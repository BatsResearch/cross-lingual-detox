import os.path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
ASSET_DIR = os.path.join(ROOT_DIR, "assets")
WEIGHT_DIR = os.path.join(ASSET_DIR, "weights")
DATASET_DIR = os.path.join(ASSET_DIR, "datasets")
