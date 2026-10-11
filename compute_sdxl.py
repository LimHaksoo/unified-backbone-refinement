import os, argparse, random, numpy as np, torch, torch.nn as nn, re
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import clip  # git+https://github.com/openai/CLIP.git
from transformers import AutoProcessor, AutoModel
from urllib.request import urlretrieve
from tqdm import tqdm
from datasets import load_dataset

"""
sdxl_eval_batch.py
"""
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

print("▶ Loading CLIP (ViT-L/14)...")
clip_model, clip_pre = clip.load("ViT-L/14", device=device)

print("▶ Loading PickScore v1 (CLIP-H)...")
PS_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PS_MODEL     = "yuvalkirstain/PickScore_v1"
pick_proc  = AutoProcessor.from_pretrained(PS_PROCESSOR)
pick_model = AutoModel.from_pretrained(PS_MODEL).eval().to(device)

# ─────────────────────────── Folder name ──────────────────────────
def safe_prompt_name(text, max_len=50):
    s = text.strip().replace('"','')
    s = re.sub(r'[\\/*?:"<>|]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s[:max_len]

# ────────────────────────── Save result ───────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
csv_path = os.path.join(args.output_dir, "scores_all.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("idx,prompt,pick_score\n")

# ───────────────────────── Image generation & Evaluation ─────────────────────────
results = []
for idx, prompt in enumerate(tqdm(prompts_list, desc="Generate & Score")):
    # 1) Image generation
    img = pipe(prompt=prompt, num_inference_steps=50, fixing=args.fixing).images[0]
    # 2) Image save
    name = safe_prompt_name(prompt)
    img_dir = os.path.join(args.output_dir, f"{idx:05d}_{name}")
    os.makedirs(img_dir, exist_ok=True)
    img.save(os.path.join(img_dir, "image.png"))
    # 3) PickScore
    img_in = clip_pre(img).unsqueeze(0).to(device)
    txt_inputs = pick_proc(text=prompt, return_tensors="pt", truncation=True).to(device)
    img_inputs = pick_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        i_feat = pick_model.get_image_features(**img_inputs)
        t_feat = pick_model.get_text_features(**txt_inputs)
        i_feat /= i_feat.norm(dim=-1, keepdim=True)
        t_feat /= t_feat.norm(dim=-1, keepdim=True)
        pick_s = (pick_model.logit_scale.exp() * (t_feat @ i_feat.T))[0][0].item()
    # 4) Record
    results.append([pick_s])
    with open(csv_path, "a", encoding="utf-8") as f:
        clean_prompt = prompt.replace('\"', '')
        f.write(f'{idx},"{clean_prompt}",{pick_s:.4f}\n')

# ────────────────────── Statistics ─────────────────────
arr = np.array(results)
mean = arr.mean(axis=0)
std  = arr.std(axis=0)
print("\n===== Summary =====")
print(f"PickScore       : {mean[0]:.2f} ± {std[0]:.2f}")
