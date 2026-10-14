"""Generate PixArt-Sigma images and evaluate PickScore."""

import argparse
import csv
from pathlib import Path
import random
import re


def safe_prompt_name(text, max_len=50):
    name = re.sub(r'[\\/*?:"<>|]', '', text.strip())
    return re.sub(r'\s+', '_', name)[:max_len]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_prompts', type=int, default=10000)
    parser.add_argument('--output_dir', type=Path, default=Path('./result_pixart'))
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument(
        '--fixing', '--suppress', dest='suppress', action='store_true',
        help='Enable internal-feature refinement.',
    )
    parser.add_argument('--q', type=float, default=0.9)
    parser.add_argument('--kappa', type=float, default=0.5)
    parser.add_argument('--layer_idx', type=int, default=23)
    args = parser.parse_args()
    if args.num_prompts <= 0 or args.steps <= 0:
        parser.error('--num_prompts and --steps must be positive')
    if not 0.0 <= args.q <= 1.0 or not 0.0 <= args.kappa <= 1.0:
        parser.error('--q and --kappa must be between 0 and 1')
    if args.device_idx < 0 or args.layer_idx < 0:
        parser.error('--device_idx and --layer_idx must be nonnegative')
    return args


def main():
    args = parse_args()
    import numpy as np
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModel, AutoProcessor

    from pixart_refinement import PixArtSigmaWrapper

    if not torch.cuda.is_available():
        raise RuntimeError('This evaluation requires an NVIDIA GPU with CUDA support.')
    device = f'cuda:{args.device_idx}'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = load_dataset('sayakpaul/pick-a-pic-v2-unique-prompts', split='train')
    prompts = dataset['prompt'][:args.num_prompts]
    if not prompts:
        raise RuntimeError('The prompt dataset is empty.')

    wrapper = PixArtSigmaWrapper(
        device=device, seed=args.seed, suppress=args.suppress, q=args.q,
        kappa=args.kappa, layer_idx=args.layer_idx, steps=args.steps,
    )
    processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    scorer = AutoModel.from_pretrained('yuvalkirstain/PickScore_v1').eval().to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'scores_all.csv'
    scores = []
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['idx', 'prompt', 'pick_score'])
        for idx, prompt in enumerate(tqdm(prompts, desc='Generate & Score')):
            image = wrapper.generate(prompt)
            image_dir = args.output_dir / f'{idx:05d}_{safe_prompt_name(prompt)}'
            image_dir.mkdir(parents=True, exist_ok=True)
            image.save(image_dir / 'image.png')

            with torch.no_grad():
                text_inputs = processor(
                    text=prompt, return_tensors='pt', truncation=True
                ).to(device)
                image_inputs = processor(images=image, return_tensors='pt').to(device)
                image_features = scorer.get_image_features(**image_inputs)
                text_features = scorer.get_text_features(**text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True) + 1e-8
                text_features /= text_features.norm(dim=-1, keepdim=True) + 1e-8
                score = (
                    scorer.logit_scale.exp() * (text_features @ image_features.T)
                )[0][0].item()
            scores.append(score)
            writer.writerow([idx, prompt, f'{score:.4f}'])
            stream.flush()

    values = np.asarray(scores, dtype=np.float32)
    print(f'PickScore: {values.mean():.2f} +/- {values.std():.2f}')
    print(f'Saved CSV: {csv_path}')


if __name__ == '__main__':
    main()
