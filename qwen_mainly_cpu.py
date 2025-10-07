import json
import os
import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image

# ================== Model setup ==================
model_id = "Qwen/Qwen-Image-Edit"
torch_dtype = torch.bfloat16
device = "cuda"

# Transformer quantization
quantization_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
).to("cpu")

# Text encoder quantization
quantization_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
).to("cpu")

# Pipeline
pipe = QwenImageEditPipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
)
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(42)

# ================== Load JSON ==================
json_path = "Dataset_full.json"
with open(json_path, "r", encoding="utf-8") as f:
    comments_data = json.load(f)

output_folder = "qwen_mainly_cpu"
os.makedirs(output_folder, exist_ok=True)

total_items = len(comments_data)

start_index = 2000
end_index = 3000

start_index = max(0, min(start_index, total_items))
end_index = max(start_index + 1, min(end_index, total_items))

num_to_process = end_index - start_index

items = comments_data[start_index:end_index]
for i, data in enumerate(items, start=1):
    current_index = start_index + i - 1
    prompt = data.get("mask_url", "")
    image_url = data.get("image_url", "")
    json_id = data.get("prefix_id", f"id_{current_index}")


    print(f"\n[{i}/{num_to_process}] Processing index {current_index} -> {json_id}")

    try:
        image = load_image(image_url).convert("RGB")
        max_size = 1024
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    except Exception as e:
        print(f"[{i}/{num_to_process}] Skipped {json_id} (cannot load image: {e})")
        continue

    try:
        edited_image = pipe(
            image,
            prompt,
            num_inference_steps=32,
            generator=generator
        ).images[0]

        output_filename = os.path.join(output_folder, f"{json_id}_output_qwen_comment_dataset.png")
        edited_image.save(output_filename)
        print(f"[{i}/{num_to_process}] Saved -> {output_filename}")

    except Exception as e:
        print(f"[{i}/{num_to_process}] Failed editing {json_id}: {e}")
        continue
