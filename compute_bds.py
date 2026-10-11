import os
import argparse
import random
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
from datasets import load_dataset
from tqdm import tqdm
import re
from PIL import ImageDraw

"""
compute_sdxl_bds_pair.py
"""

# ────────────────────────── Parsing ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device_idx",  type=int,   default=0)
parser.add_argument("--num_prompts", type=int,   default=5000)
parser.add_argument("--output_dir",  type=str,   required=True,
    help="Image directory (Ex: ./result_batch)")
parser.add_argument("--csv_path",    type=str,   default="scores_bds_pair.csv",
    help="BDS CSV name to be saved")
args = parser.parse_args()

# ─────────────────────────── Seed & Device ─────────────────────────
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
device = f"cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────── Prompt load ───────────────────────────
human_keywords = ["person","woman","man","girl","boy","portrait","face","selfie","body"]
ds = load_dataset("sayakpaul/pick-a-pic-v2-unique-prompts", split="train")
prompts = ds["prompt"][:args.num_prompts]

exception_indices = []
prompts_list = [
    (idx, p) for idx, p in enumerate(prompts, start=0)
    if (idx) not in exception_indices and any(k in p.lower() for k in human_keywords)
]
print(f"▶ # prompts related to Human: {len(prompts_list)}")

# ─────────────────────────── YOLOv8-Pose load ─────────────────────
print("▶ Loading YOLOv8-Pose model…")
model = YOLO("yolov8n-pose.pt")

# COCO keypoint indices (17 KP + 8 limb)
LIMBS = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]
DENOM = 17 + len(LIMBS)

def safe_name(text, max_len=50):
    s = text.strip().replace('"','')
    s = re.sub(r'[\\/*?:"<>|]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s[:max_len]

def collect_keypoints(img: Image.Image, conf_thr: float = 0.2):
    """YOLOv8-Pose → (xy, conf, valid_mask) or (None, None, None) on failure."""
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    res   = model.predict(frame, device=device, imgsz=640, conf=0.1, verbose=False)
    if not res or res[0].keypoints is None or res[0].keypoints.data.shape[1] == 0:
        return None, None, None
    arr   = res[0].keypoints.data[0].cpu().numpy()  # (17,3)
    xy    = arr[:, :2]
    cf    = arr[:,  2]
    valid = cf >= conf_thr
    return xy, cf, valid

def bds_pair(orig_img, fix_img, conf_thr: float = 0.2, check: bool = False):
    """Calculate BDS based on detected more"""
    xy_f, cf_f, v_f = collect_keypoints(fix_img, conf_thr)
    xy_o, cf_o, v_o = collect_keypoints(orig_img, conf_thr)


    if check:
        assert idx is not None, "idx must be provided when check=True"
        save_dir = os.path.join('./comparisons_bds_check', f"{idx:05d}")
        os.makedirs(save_dir, exist_ok=True)

        # --- Plot keypoints ---
        vis_o = orig_img.copy()
        draw_o = ImageDraw.Draw(vis_o)
        if xy_o is not None:
            for (x, y), conf, valid in zip(xy_o, cf_o, v_o):
                if valid:
                    r = int(2 + conf * 8)
                    draw_o.ellipse(
                        [(x - r, y - r), (x + r, y + r)],
                        outline="red", width=2
                    )
        vis_o.save(os.path.join(save_dir, "orig_keypoints.png"))

        vis_f = fix_img.copy()
        draw_f = ImageDraw.Draw(vis_f)
        if xy_f is not None:
            for (x, y), conf, valid in zip(xy_f, cf_f, v_f):
                if valid:
                    r = int(2 + conf * 8)
                    draw_f.ellipse(
                        [(x - r, y - r), (x + r, y + r)],
                        outline="blue", width=2
                    )
        vis_f.save(os.path.join(save_dir, "fixed_keypoints.png"))

    if v_f is None and v_o is None:
        return 1.0, 1.0

    # None 처리
    if v_f is None:
        xy_f = np.zeros((17,2))
        cf_f = np.zeros(17)
        v_f  = np.zeros(17, dtype=bool)

    if v_o is None:
        xy_o = np.zeros((17,2))
        cf_o = np.zeros(17)
        v_o  = np.zeros(17, dtype=bool)

    V = v_f if v_f.sum() >= v_o.sum() else v_o

    # torso (5,6 shoulder)
    if not (V[5] and V[6]):
        return 1.0, 1.0
    torso = np.linalg.norm(xy_f[5] - xy_f[6]) + 1e-6

    def count_missing(v):
        return int((~v & V).sum())

    def count_bad(xy, valid):
        bad = 0
        for a, b in LIMBS:
            if not (V[a] and V[b]):
                continue
            if not (valid[a] and valid[b]):
                bad += 1
            else:
                length = np.linalg.norm(xy[a] - xy[b])
                if length < 0.3 * torso or length > 1.5 * torso:
                    bad += 1
        return bad

    missing_o = count_missing(v_o)
    missing_f = count_missing(v_f)
    bad_o     = count_bad(xy_o, v_o)
    bad_f     = count_bad(xy_f, v_f)

    return (missing_o + bad_o) / DENOM, (missing_f + bad_f) / DENOM

# ───────────────────── Save results ────────────────────────────
os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
with open(args.csv_path, "w", encoding="utf-8") as f:
    f.write("idx,prompt,bds_original,bds_fixed\n")

scores_o, scores_f = [], []

print("▶ Calculating paired BDS for each prompt…")
for idx, prompt in tqdm(prompts_list, desc="BDS Pair Compute"):
    
    name      = safe_name(prompt)
    dir_orig  = os.path.join(args.output_dir, "original", f"{idx:05d}_{name}")
    dir_fixed = os.path.join(args.output_dir, "fixed",    f"{idx:05d}_{name}")
    p_orig    = os.path.join(dir_orig,  "image.png")
    p_fixed   = os.path.join(dir_fixed, "image.png")

    if not os.path.exists(p_orig) or not os.path.exists(p_fixed):
        print(f"[WARN] Missing image idx={idx}")
        continue

    img_o = Image.open(p_orig).convert("RGB")
    img_f = Image.open(p_fixed).convert("RGB")
    b_o, b_f = bds_pair(img_o, img_f, check=True)

    scores_o.append(b_o)
    scores_f.append(b_f)
    with open(args.csv_path, "a", encoding="utf-8") as f:
        safe_p = prompt.replace('"', "")
        f.write(f'{idx},"{safe_p}",{b_o:.4f},{b_f:.4f}\n')

# ───────────────────── Statistic ─────────────────────────────
arr_o, arr_f = np.array(scores_o), np.array(scores_f)
m_o, s_o     = arr_o.mean(), arr_o.std()
m_f, s_f     = arr_f.mean(), arr_f.std()

print(f"\n===== Paired BDS Summary (N={len(scores_o)}) =====")
print(f"\nPath    : {args.output_dir}")
print(f"Original BDS    : {m_o:.4f} ± {s_o:.4f}")
print(f"Fixed    BDS    : {m_f:.4f} ± {s_f:.4f}")
