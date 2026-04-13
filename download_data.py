import traceback
import urllib.request
import urllib.error
import soundata
import soundata.download_utils as download_utils
from src.config import ROOT_DIR, RAW_DATA_DIR

download_utils.urllib.request = urllib.request
download_utils.urllib.error = urllib.error

def main():
    print("Start downloading UrbanSound8K...")
    print("Project root:", ROOT_DIR)
    print("Save directory:", RAW_DATA_DIR)

    dataset = soundata.initialize("urbansound8k", data_home=str(RAW_DATA_DIR))
    dataset.download()
    dataset.validate()

    print("Download completed.")
    print("Dataset root:", RAW_DATA_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Download failed.")
        print(type(e).__name__, str(e))
        traceback.print_exc()
