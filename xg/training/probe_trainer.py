import gc
import logging
import os
import os.path as osp
import random
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from xg import DATASET_DIR, WEIGHT_DIR
from xg.probe import Discriminator

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")


class CSVDataset(Dataset):

    def __init__(self, csv_file: str):

        # Load csv file
        self.data_frame = pd.read_csv(csv_file)

        # Sanity check
        if (
            "text" not in self.data_frame.columns
            or "label" not in self.data_frame.columns
        ):
            raise RuntimeError("Invalid dataset loaded!!")

        if not self.data_frame["label"].isin([0, 1]).all():
            raise RuntimeError(
                "Invalid dataset: label can only be 0 or 1 (not toxic vs. toxic)"
            )

        # Logging
        positive_count = self.data_frame["label"].sum()
        negative_count = len(self.data_frame) - positive_count

        info = f"Successfully loaded {len(self.data_frame)} samples: {positive_count} toxic sentences, {negative_count} non-toxic sentences."
        logging.info(info)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.data_frame.iloc[idx]["text"]
        label = self.data_frame.iloc[idx]["label"]
        return sentence, label


def collate_fn(batch):
    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return sentences, torch.tensor(labels)


def set_global_seed(seed: int):
    """Fix seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainArgs:
    model_name: str
    train_fp: str
    pooling_type: str = "avg"
    learning_rate: float = 0.0001
    val_fp: str | None = None
    epochs: int = 10
    batch_size: int = 3
    output_fp: str | None = None
    save_model: bool = True
    device: str = "cuda"
    seed: int = 99
    use_absolute_path: bool = False
    log_wandb: bool = True
    wandb_project_name: str | None = None

    def __post_init__(self):

        # Not saving output if no path provided
        if not self.output_fp:
            self.save_model = False
            logging.info("No output file path provided, not saving model")

        if not self.use_absolute_path:
            if DATASET_DIR not in self.train_fp:
                self.train_fp = osp.join(DATASET_DIR, self.train_fp)
            if self.output_fp and WEIGHT_DIR not in self.output_fp:
                self.output_fp = osp.join(WEIGHT_DIR, self.output_fp)

        # Warn if no validation dataset provided
        if not self.val_fp:
            logging.info("No validation dataset provided.")

        # Not logging if no project_name provided
        if not self.wandb_project_name:
            self.log_wandb = False

        pprint(self)

    def to_dict(self) -> Dict:
        return asdict(self)


class Trainer:
    """Trainer class"""

    def __init__(self, args: TrainArgs):
        self.args = args
        self.model = Discriminator(
            pretrained_model=args.model_name,
            pooling_type=args.pooling_type,
            device=args.device,
        )

        if args.save_model:
            if not osp.exists(args.output_fp):
                os.makedirs(args.output_fp)
        dataset = CSVDataset(csv_file=args.train_fp)

        # Init WandB
        if args.log_wandb:
            wandb.init(project=args.wandb_project_name, config=args.to_dict())

        # Set seed
        set_global_seed(args.seed)

        # Load dataset
        self.dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )

        # Validation dataset
        if args.val_fp:
            self.validation_dataloader = DataLoader(
                CSVDataset(csv_file=args.val_fp),
                batch_size=args.batch_size,
                collate_fn=collate_fn,
            )
        else:
            logging.info(
                "No validation dataset given, splitting training dataset to 90:10 for validation"
            )
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            self.dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            self.validation_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, collate_fn=collate_fn
            )

        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.BCELoss()

    def train(self) -> None:
        best_model = None
        best_epoch = -1
        best_loss = torch.inf

        for epoch in range(self.args.epochs):
            total_loss = 0
            total_gradient_step = 0
            for texts, labels in tqdm(
                self.dataloader, desc=f"Training - Epoch {epoch + 1}: "
            ):

                # Forward
                labels = labels.float().to(self.args.device)
                self.optimizer.zero_grad()
                outputs = self.model(texts)
                loss = self.criterion(outputs.squeeze(), labels)

                # Backward
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_gradient_step += 1
                total_loss += loss.item()

            # Train loss full batch
            avg_training_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1} loss: {avg_training_loss}")
            if self.args.log_wandb:
                wandb.log({"Average Training Loss": avg_training_loss})

            # Validation
            validation_loss = self.validate()

            # Save model every epoch
            if self.args.save_model:
                model_name = f"probe-ep{epoch+1}.pt"
                model_path = osp.join(self.args.output_fp, model_name)
                probe = self.model.probe_model
                torch.save(probe, model_path)

                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_epoch = epoch
                    best_model = probe

        # Save best model
        if self.args.save_model:
            model_name = "best-probe.pt"
            model_path = osp.join(self.args.output_fp, model_name)
            probe = best_model
            torch.save(probe, model_path)
            logging.info(
                f"Saving the best model from Ep {best_epoch + 1} with validation loss: {best_loss}"
            )

    def validate(self) -> float:

        total_val_loss = 0
        total_predictions = 0
        correct_predictions = 0

        with torch.no_grad():
            for texts, labels in tqdm(self.validation_dataloader, desc="Validation: "):

                # Forward
                labels = labels.float().to(self.args.device)
                outputs = self.model(texts)

                # Validation loss
                val_loss = self.criterion(outputs.squeeze(), labels)
                total_val_loss += val_loss.item()

                # Validation accuracy per batch
                preds = (outputs > 0.5).long()
                correct_predictions += (preds.squeeze() == labels).sum().item()
                total_predictions += labels.size(0)

            # Validation stats whole batch
            avg_val_loss = total_val_loss / len(self.validation_dataloader)

            final_accuracy = correct_predictions / total_predictions

        if self.args.log_wandb:
            wandb.log(
                {
                    "Average Validation Loss": avg_val_loss,
                    "Validation Accuracy": final_accuracy,
                }
            )
        return avg_val_loss
