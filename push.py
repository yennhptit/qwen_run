from huggingface_hub import HfApi, HfFolder
import os
import itertools
import time
from tqdm import tqdm
import shutil
import os


HF_TOKEN = "........"   # 
REPO_ID = "yuu1234/Dataset_Qwen_Comemnt"
LOCAL_DIR = "/disk/yuu/dataset/qwen_comment/0-10000"
TARGET_FOLDER = os.path.basename(os.path.normpath(LOCAL_DIR))
BRANCH = "main"
BATCH_SIZE = 500  # number of files per upload batch
DELETE_LOCAL = True  # delete local files after upload

HfFolder.save_token(HF_TOKEN)
api = HfApi()

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def safe_remove(path):
    try:
        os.remove(path)
    except Exception as e:
        print(f"Can't remove {path}: {e}")

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
files = [os.path.join(LOCAL_DIR, f)
         for f in os.listdir(LOCAL_DIR)
         if f.lower().endswith(image_extensions)]

print(f"Find {len(files)} images in {LOCAL_DIR}")

for batch_id, batch_files in enumerate(chunk_list(files, BATCH_SIZE), start=1):
    print(f"\nUpload batch {batch_id} ({len(batch_files)} images)...")

    # Create temporary directory for batch
    temp_dir = os.path.join(LOCAL_DIR, f"_batch_{batch_id}")
    os.makedirs(temp_dir, exist_ok=True)

    # Copy files to temporary directory
    for f in batch_files:
        shutil.copy(f, temp_dir)

    try:
        api.upload_folder(
            folder_path=temp_dir,
            path_in_repo=TARGET_FOLDER,
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=BRANCH,
            token=HF_TOKEN,
            commit_message=f"Upload batch {batch_id} ({len(batch_files)} files) → {TARGET_FOLDER}",
        )
        print(f"Batch {batch_id} upload completed.")

        if DELETE_LOCAL:
            for f in batch_files:
                safe_remove(f)

    except Exception as e:
        print(f"Error uploading batch {batch_id}: {e}")
        time.sleep(5)

    # Remove temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Finished batch {batch_id}.\n")