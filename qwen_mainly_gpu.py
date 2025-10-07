import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from PIL import Image

# ================== Model setup ==================
model_id = "Qwen/Qwen-Image-Edit"
device = "cuda"
torch_dtype = torch.float16  # Use float16 for better GPU efficiency

print("🚀 Loading Qwen-Image-Edit model on GPU...")

# Load transformer and text encoder directly on GPU
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch_dtype,
).to(device)

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    torch_dtype=torch_dtype,
).to(device)

# Create pipeline
pipe = QwenImageEditPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch_dtype,
).to(device)

# Load LoRA weights (Lightning)
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
)

# Everything stays on GPU
generator = torch.Generator(device=device).manual_seed(42)

# ================== Load JSON ==================
json_path = "Dataset_full.json"
with open(json_path, "r", encoding="utf-8") as f:
    comments_data = json.load(f)

output_folder = "qwen_full_gpu"
os.makedirs(output_folder, exist_ok=True)

total_items = len(comments_data)
start_index = 2000
end_index = 3000

start_index = max(0, min(start_index, total_items))
end_index = max(start_index + 1, min(end_index, total_items))
num_to_process = end_index - start_index

items = comments_data[start_index:end_index]

print(f"Processing {num_to_process} images from index {start_index} to {end_index-1}...")

for i, data in enumerate(items, start=1):
    current_index = start_index + i - 1
    prompt = data.get("mask_url", "")
    image_url = data.get("image_url", "")
    json_id = data.get("prefix_id", f"id_{current_index}")

    print(f"\n[{i}/{num_to_process}] Processing {json_id}")

    try:
        # Load and resize image
        image = load_image(image_url).convert("RGB")
        max_size = 1024
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    except Exception as e:
        print(f"[{i}/{num_to_process}] Skipped {json_id} (cannot load image: {e})")
        continue

    try:
        # Run image editing pipeline
        edited_image = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=32,
            generator=generator
        ).images[0]

        # Save output image
        output_filename = os.path.join(output_folder, f"{json_id}_output_qwen_comment_dataset.png")
        edited_image.save(output_filename)
        print(f"[{i}/{num_to_process}] Saved -> {output_filename}")

    except Exception as e:
        print(f"[{i}/{num_to_process}] Failed to edit {json_id}: {e}")
        continue

print("\n Done! All successfully processed images are saved in:", output_folder)
