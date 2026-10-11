# Unified Backbone Refinement for Diffusion Models via Internal-Latent Analysis

SDXL evaluation code accompanying **Unified Backbone Refinement for Diffusion Models via Internal-Latent Analysis**.

## Setup

Use a dedicated Python environment with an NVIDIA GPU and CUDA support. The supplied requirements pin PyTorch 2.3.0 with CUDA 12.1 and Diffusers 0.33.1.

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Replace these files in the installed Diffusers package with the corresponding files from this repository:

| Repository file | Destination inside the installed `diffusers` package |
| --- | --- |
| `pipeline_stable_diffusion_xl.py` | `pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py` |
| `unet_2d_condition.py` | `models/unets/unet_2d_condition.py` |
| `scheduling_dpmsolver_multistep.py` | `schedulers/scheduling_dpmsolver_multistep.py` |

Locate the installed package with:

```bash
python -c "import diffusers; print(diffusers.__path__[0])"
```

## Generate images and compute PickScore

Run the original and refined versions with the same seed and prompt count. The `--fixing` flag enables refinement.

```bash
python compute_sdxl.py --device_idx 0 --seed 0 --num_prompts 100 --output_dir ./results/original
python compute_sdxl.py --device_idx 0 --seed 0 --num_prompts 100 --output_dir ./results/fixed --fixing
```

Each run saves images and a `scores_all.csv` file in its output directory. The scripts download the required models and prompts on first use.

## Compute paired Body Distortion Score (BDS)

Pass the parent directory containing both `original/` and `fixed/`:

```bash
python compute_bds.py --num_prompts 100 --output_dir ./results --csv_path ./results/scores_bds_pair.csv
```

The script evaluates prompts matching its human-related keyword filter and saves keypoint visualizations in `comparisons_bds_check/`.

## Third-party code

The supplied Diffusers files retain their original copyright and Apache 2.0 license headers. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
