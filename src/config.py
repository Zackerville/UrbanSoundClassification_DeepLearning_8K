from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
AUDIO_DIR = RAW_DATA_DIR / "audio"
METADATA_PATH = ROOT_DIR / "data" / "metadata" / "UrbanSound8K.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
EDA_DIR = OUTPUT_DIR / "eda"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 4.0
TARGET_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
TOP_DB = 80

NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_WORKERS = 2

TEST_FOLD = 1
VAL_FOLD = 2

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]