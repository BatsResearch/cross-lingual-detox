from xg.training.probe_trainer import TrainArgs, Trainer


def main():

    args = TrainArgs(
        model_name="ai-forever/mGPT",
        train_fp=...,  # substitue with Jigsaw english dataset
        batch_size=10,
        epochs=20,
        output_fp="english_probe",
        learning_rate=1e-6,
        # wandb_project_name="english_prob",  # Uncomment to enable wandb
    )

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
