"""Generate SDXL images with optional internal-feature refinement."""

import os, argparse, random, numpy as np, torch, re
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
from datasets import load_dataset

# ─────────────────────────────── Parsing ───────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device_idx", type=int, default=0)
parser.add_argument("--seed",       type=int, default=0)
parser.add_argument("--num_prompts",type=int, default=10000,
                    help="Number of prompts to load from Pick-a-Pic")
parser.add_argument("--output_dir", type=str, default="./result_batch")
parser.add_argument("--fixing", action="store_true")
args = parser.parse_args()

# ───────────────────────────── Seed & Device ───────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = f"cuda:{args.device_idx}"

# ───────────────────────────── prompts load ─────────────────────────
print(f"▶ Loading {args.num_prompts} prompts from Pick-a-Pic...")
dataset = load_dataset(
    "sayakpaul/pick-a-pic-v2-unique-prompts", split="train"
)
prompts_list = dataset["prompt"][ : args.num_prompts ] 

# ────────────────────────── Model load ─────────────────────────
print("▶ Loading SD-XL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# ─────────────────────────── Folder name ──────────────────────────
def safe_prompt_name(text, max_len=50):
    s = text.strip().replace('"','')
    s = re.sub(r'[\\/*?:"<>|]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s[:max_len]

# ────────────────────────── Save result ───────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
# ───────────────────────── Image generation ─────────────────────────
for idx, prompt in enumerate(tqdm(prompts_list, desc="Generate")):
    # 1) Image generation
    img = pipe(prompt=prompt, num_inference_steps=50, fixing=args.fixing).images[0]
    # 2) Image save
    name = safe_prompt_name(prompt)
    img_dir = os.path.join(args.output_dir, f"{idx:05d}_{name}")
    os.makedirs(img_dir, exist_ok=True)
    img.save(os.path.join(img_dir, "image.png"))

print(f"Saved images: {args.output_dir}")
