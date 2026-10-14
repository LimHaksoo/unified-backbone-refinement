# Unified Backbone Refinement for Diffusion Models via Internal-Latent Analysis

SDXL and PixArt-Sigma refinement and image generation code accompanying **Unified Backbone Refinement for Diffusion Models via Internal-Latent Analysis**.

## SDXL setup

Use a dedicated Python environment with an NVIDIA GPU and CUDA support. The supplied requirements pin PyTorch 2.3.0 with CUDA 12.1 and Diffusers 0.33.1.

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

For SDXL, replace these files in the installed Diffusers package with the corresponding files from this repository. PixArt-Sigma uses the standard Diffusers 0.33.1 pipeline and the standalone hook in `pixart_refinement.py`; it does not require these replacements.

| Repository file | Destination inside the installed `diffusers` package |
| --- | --- |
| `pipeline_stable_diffusion_xl.py` | `pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py` |
| `unet_2d_condition.py` | `models/unets/unet_2d_condition.py` |
| `scheduling_dpmsolver_multistep.py` | `schedulers/scheduling_dpmsolver_multistep.py` |

Locate the installed package with:

```bash
python -c "import diffusers; print(diffusers.__path__[0])"
```

## SDXL: generate images

Run the original and refined versions with the same seed and prompt count. The `--fixing` flag enables refinement.

```bash
python compute_sdxl.py --device_idx 0 --seed 0 --num_prompts 100 --output_dir ./results/original
python compute_sdxl.py --device_idx 0 --seed 0 --num_prompts 100 --output_dir ./results/fixed --fixing
```

Each run saves images in its output directory. The scripts download the required models and prompts on first use.

## PixArt-Sigma: generate images

In a separate Python 3.10 environment with an NVIDIA GPU and CUDA support, install the PixArt requirements:

```bash
python -m pip install -r requirements_pixart.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The PixArt requirements use NumPy 1.26.4 for compatibility with the pinned PyTorch 2.3 build. No Diffusers source-file replacement is needed.

The PixArt implementation uses `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`. Run both versions with the same seed and prompt count:

```bash
python compute_pixart.py --device_idx 0 --seed 0 --num_prompts 100 --output_dir ./results/pixart/original
python compute_pixart.py --device_idx 0 --seed 0 --num_prompts 100 --output_dir ./results/pixart/fixed --fixing
```

`--fixing` (also accepted as `--suppress`) enables refinement. Defaults from the supplied implementation are 50 inference steps, block index 23 (zero-based), quantile `q=0.9`, blending coefficient `kappa=0.5`, and EMA coefficient 0.7. Refinement is applied at zero-based steps 5 through 9. The command-line options `--steps`, `--layer_idx`, `--q`, and `--kappa` expose the corresponding settings. The generator seed is reset for each prompt.

Each command saves images in prompt-specific subdirectories of its `--output_dir`. The required model weights and prompts are downloaded on first use.

## Third-party code

The supplied Diffusers files retain their original copyright and Apache 2.0 license headers. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
