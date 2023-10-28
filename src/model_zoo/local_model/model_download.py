import argparse
from transformers import snapshot_download
import time
import logging

# Set up logging
logging.basicConfig(filename='download_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_with_retry(repo_id, max_retries=20, retry_interval=1):
    for attempt in range(max_retries):
        try:
            x = snapshot_download(repo_id=repo_id, local_files_only=False, resume_download=True)
            message = f"Download successful on attempt {attempt + 1}: {x}"
            logging.info(message)
            print(message)
            return x  # Download successful, return the result
        except Exception as e:
            message = f"Download failed on attempt {attempt + 1}: {e}"
            logging.error(message)
            print(message)
            print("Retrying in {} seconds...".format(retry_interval))
            time.sleep(retry_interval)

    return None  # Download failed after all retries


def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("model_name", type=str, help="Name of the model to download.")
    args = parser.parse_args()

    model_name = args.model_name

    message = f"Starting download of model: {model_name}"
    logging.info(message)
    print(message)

    x = download_with_retry(repo_id=model_name)

    if x is not None:
        message = "Download successful: {}".format(x)
        logging.info(message)
        print(message)
    else:
        message = "Download failed after multiple attempts."
        logging.error(message)
        print(message)

    message = "Download process completed."
    logging.info(message)
    print(message)


if __name__ == "__main__":
    main()
