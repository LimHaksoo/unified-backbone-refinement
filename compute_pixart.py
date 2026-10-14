"""Generate PixArt-Sigma images with optional internal-feature refinement."""

import argparse
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

    from pixart_refinement import PixArtSigmaWrapper

    if not torch.cuda.is_available():
        raise RuntimeError('Image generation requires an NVIDIA GPU with CUDA support.')
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, prompt in enumerate(tqdm(prompts, desc='Generate')):
        image = wrapper.generate(prompt)
        image_dir = args.output_dir / f'{idx:05d}_{safe_prompt_name(prompt)}'
        image_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_dir / 'image.png')

    print(f'Saved images: {args.output_dir}')


if __name__ == '__main__':
    main()
