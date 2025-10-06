import json
import os
import torch
import subprocess
import time
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image

# ================== CONFIG ==================
MODEL_ID = "Qwen/Qwen-Image-Edit"
JSON_PATH = "Dataset_full.json"
OUTPUT_FOLDER = "qwen_mainly_gpu"
HF_REPO = "yuu1234/Dataset_Qwen_Comemnt" 
BATCH_SIZE = 1                                # Number of images to edit at once
TORCH_DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_INDEX, END_INDEX = 2000, None          
# ============================================

# ================== FUNCTIONS ==================
def push_to_huggingface(retries=3):
    """Push current batch to Hugging Face and clear local folder."""
    if not os.listdir(OUTPUT_FOLDER):
        return  # nothing to push

    print("\n Uploading batch to Hugging Face Hub...")
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(
                ["hf", "upload", HF_REPO, OUTPUT_FOLDER, "--repo-type=dataset"],
                check=True
            )
            print("Upload successful!")

            # Clear local folder
            for file in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(" Local folder cleared.\n")
            return
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                print("Retrying in 10s...")
                time.sleep(10)
            else:
                print("Upload failed after all retries.\n")

# ================== MODEL SETUP ==================
print("Loading model components...")

# Transformer quantization
transformer_quant = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=TORCH_DTYPE,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", quantization_config=transformer_quant, torch_dtype=TORCH_DTYPE
).to(DEVICE)

# Text encoder quantization
text_quant = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=TORCH_DTYPE,
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, subfolder="text_encoder", quantization_config=text_quant, torch_dtype=TORCH_DTYPE
).to(DEVICE)

# Pipeline setup
pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID, transformer=transformer, text_encoder=text_encoder, torch_dtype=TORCH_DTYPE
)
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
)
pipe.to(DEVICE)
generator = torch.Generator(device=DEVICE).manual_seed(42)
print("Model loaded successfully!\n")

# ================== LOAD DATA ==================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(JSON_PATH, "r", encoding="utf-8") as f:
    comments_data = json.load(f)

total_items = len(comments_data)
start_index = max(0, min(START_INDEX, total_items))
end_index = END_INDEX if END_INDEX is not None else total_items
items = comments_data[start_index:end_index]
print(f"Processing {len(items)} images (index {start_index}–{end_index-1})")

# ================== MAIN LOOP ==================
batch_images, batch_prompts, batch_ids = [], [], []

for i, data in enumerate(items, start=1):
    current_index = start_index + i - 1
    prompt = data.get("mask_url", "")
    image_url = data.get("image_url", "")
    json_id = data.get("prefix_id", f"id_{current_index}")

    print(f"\n[{i}/{len(items)}] Processing index {current_index} -> {json_id}")

    try:
        image = load_image(image_url).convert("RGB")
        image.thumbnail((1024, 1024), Image.LANCZOS)
        batch_images.append(image)
        batch_prompts.append(prompt)
        batch_ids.append(json_id)
    except Exception as e:
        print(f" Skipped {json_id} (cannot load image: {e})")
        continue

    # If batch is full or last image
    if len(batch_images) == BATCH_SIZE or i == len(items):
        try:
            edited_images = pipe(batch_images, batch_prompts, num_inference_steps=32, generator=generator).images
            for img, img_id in zip(edited_images, batch_ids):
                output_file = os.path.join(OUTPUT_FOLDER, f"{img_id}_output.png")
                img.save(output_file)
                print(f"Saved -> {output_file}")
        except Exception as e:
            print(f"Failed editing batch: {e}")

        # Push and clear
        push_to_huggingface()

        # Reset batch
        batch_images, batch_prompts, batch_ids = [], [], []

print("All done!")
