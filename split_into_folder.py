import os
import json
import shutil

json_path = "Dataset_full.json"
image_dir = "/disk/yuu/dataset/qwen_comment"

GROUP_SIZE = 10000

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

prefix_to_index = {item["prefix_id"]: item["index"] for item in data}

def get_range_folder(index):
    if index <= GROUP_SIZE:
        #0-10000
        return f"0-{GROUP_SIZE}"
    else:
        #10001–20000, 20001–30000, ...
        group_start = ((index - 1) // GROUP_SIZE) * GROUP_SIZE + 1
        group_end = group_start + GROUP_SIZE - 1
        return f"{group_start}-{group_end}"

for filename in os.listdir(image_dir):
    if filename.endswith("_output_qwen_comment_dataset.png"):
        prefix_id = filename.split("_output_qwen_comment_dataset.png")[0]

        if prefix_id in prefix_to_index:
            index = prefix_to_index[prefix_id]
            subfolder = get_range_folder(index)

            dest_folder = os.path.join(image_dir, subfolder)
            os.makedirs(dest_folder, exist_ok=True)

            src = os.path.join(image_dir, filename)
            dst = os.path.join(dest_folder, filename)

            shutil.move(src, dst)
            print(f"{filename} → {subfolder}/")
        else:
            print(f"Can't find prefix_id in JSON: {prefix_id}")
