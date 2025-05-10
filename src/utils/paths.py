import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
SRC_DIR = pathlib.Path(__file__).parent.parent
HEATMAPS_DIR = ROOT_DIR / "heatmaps"
QTABLE_DIR = ROOT_DIR/ "qtables"
IMAGE_DIR = SRC_DIR / "images"
WEIGHTS_DIR = SRC_DIR / "weights"
EPISODES_DIR = SRC_DIR / "episodes"
CONFIG_DIR = SRC_DIR / "configs"
