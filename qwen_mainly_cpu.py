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
OUTPUT_FOLDER = "qwen_mainly_cpu"
HF_REPO = "yuu1234/Dataset_Qwen_Comemnt" 
UPLOAD_BATCH_SIZE = 5                    
TORCH_DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================


# ================== FUNCTIONS ==================
def push_to_huggingface(retries=3):
    """Push current batch to Hugging Face and clear local folder."""
    print("\nUploading batch to Hugging Face Hub...")

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "hf", "upload",
                    HF_REPO,
                    OUTPUT_FOLDER,
                    "--repo-type=dataset",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print("Upload successful!")

            # Xóa toàn bộ file trong folder sau khi upload
            for file in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Local folder cleared.\n")
            return

        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt}/{retries} failed:\n{e.stderr.strip()}")
            if attempt < retries:
                print("Retrying in 10 seconds...")
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
    MODEL_ID,
    subfolder="transformer",
    quantization_config=transformer_quant,
    torch_dtype=TORCH_DTYPE,
).to("cpu")

# Text encoder quantization
text_quant = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=TORCH_DTYPE,
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    subfolder="text_encoder",
    quantization_config=text_quant,
    torch_dtype=TORCH_DTYPE,
).to("cpu")

# Pipeline setup
pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID, transformer=transformer, text_encoder=text_encoder, torch_dtype=TORCH_DTYPE
)
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
)
pipe.enable_model_cpu_offload()
generator = torch.Generator(device=DEVICE).manual_seed(42)
print("Model loaded successfully!\n")


# ================== LOAD DATA ==================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(JSON_PATH, "r", encoding="utf-8") as f:
    comments_data = json.load(f)

total_items = len(comments_data)
start_index, end_index = 2000, None
start_index = max(0, min(start_index, total_items))
end_index = max(start_index + 1, min(end_index, total_items))
items = comments_data[start_index:end_index]

print(f"Processing {len(items)} images (index {start_index}–{end_index - 1})")

# ================== MAIN LOOP ==================
saved_count = 0

for i, data in enumerate(items, start=1):
    current_index = start_index + i - 1
    prompt = data.get("mask_url", "")
    image_url = data.get("image_url", "")
    json_id = data.get("prefix_id", f"id_{current_index}")

    print(f"\n[{i}/{len(items)}] Processing index {current_index} -> {json_id}")

    try:
        image = load_image(image_url).convert("RGB")
        max_size = 1024
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    except Exception as e:
        print(f"Skipped {json_id} (cannot load image: {e})")
        continue

    try:
        edited_image = pipe(
            image,
            prompt,
            num_inference_steps=32,
            generator=generator
        ).images[0]

        output_filename = os.path.join(OUTPUT_FOLDER, f"{json_id}_output_qwen_comment.png")
        edited_image.save(output_filename)
        print(f"Saved -> {output_filename}")

        saved_count += 1

        # === Push batch every N images ===
        if saved_count >= UPLOAD_BATCH_SIZE:
            push_to_huggingface()
            saved_count = 0

    except Exception as e:
        print(f"Failed editing {json_id}: {e}")
        continue

# Push remaining images
if saved_count > 0:
    push_to_huggingface()

print("All done!")
