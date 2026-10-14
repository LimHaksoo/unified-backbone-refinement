"""PixArt-Sigma generation with internal-feature refinement."""

import torch


class AnomalySuppressor:
    """Refine a selected Transformer block using its score-space EMA."""

    def __init__(self, pipeline, layer_idx=23, q=0.9, kappa=0.5, ema_beta=0.7):
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be between 0 and 1")
        if not 0.0 <= kappa <= 1.0:
            raise ValueError("kappa must be between 0 and 1")
        if not 0.0 <= ema_beta <= 1.0:
            raise ValueError("ema_beta must be between 0 and 1")
        if not 0 <= layer_idx < len(pipeline.transformer.transformer_blocks):
            raise ValueError("layer_idx is outside the Transformer block range")
        self.pipeline = pipeline
        self.layer_idx = layer_idx
        self.q = float(q)
        self.kappa = float(kappa)
        self.beta = float(ema_beta)
        self.cur_step = 0
        self.ema = None
        self.handle = None

    def set_step(self, step_idx):
        self.cur_step = int(step_idx)

    @torch.no_grad()
    def _hook(self, module, inputs, output):
        sigma = float(self.pipeline.scheduler.sigmas[self.cur_step].item())
        score = -output.float() / (sigma + 1e-12)
        refined = output

        # Preserve the original implementation's fixed, zero-based step window.
        if self.ema is not None and 5 <= self.cur_step < 10:
            log_score = torch.log1p(score.abs() + 1e-6)
            log_ema = torch.log1p(self.ema.float().abs() + 1e-6)
            difference = (log_score - log_ema).abs()
            threshold = torch.quantile(
                difference.view(difference.size(0), -1), self.q, dim=1, keepdim=True
            )
            mask = difference > threshold.view(-1, 1, 1)
            reference = self.ema.float().clone()
            reference *= -sigma
            refined = output.float() * (~mask) + (
                (1.0 - self.kappa) * reference + self.kappa * output.float()
            ) * mask

        if self.ema is None:
            self.ema = score.clone()
        else:
            self.ema = self.beta * self.ema + (1.0 - self.beta) * score
        return refined.to(output.dtype)

    def install(self):
        if self.handle is not None:
            raise RuntimeError("Refinement hook is already installed")
        block = self.pipeline.transformer.transformer_blocks[self.layer_idx]
        self.handle = block.register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class PixArtSigmaWrapper:
    """Standalone wrapper for the supplied PixArt-Sigma refinement code."""

    def __init__(
        self,
        device="cuda:0",
        seed=0,
        suppress=False,
        q=0.9,
        kappa=0.5,
        layer_idx=23,
        steps=50,
    ):
        from diffusers import PixArtSigmaPipeline

        if steps <= 0:
            raise ValueError("steps must be positive")
        if not 0.0 <= q <= 1.0 or not 0.0 <= kappa <= 1.0:
            raise ValueError("q and kappa must be between 0 and 1")
        if layer_idx < 0:
            raise ValueError("layer_idx must be nonnegative")
        self.seed = seed
        self.suppress = suppress
        self.q = q
        self.kappa = kappa
        self.layer_idx = layer_idx
        self.num_inf_steps = steps
        self.model = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        ).to(device)
        self.generator = torch.Generator(device=device)

    @torch.no_grad()
    def generate(self, caption):
        # Reset per prompt, matching the original evaluation wrapper.
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        if not self.suppress:
            return self.model(
                prompt=caption,
                num_inference_steps=self.num_inf_steps,
                generator=self.generator,
            ).images[0]

        suppressor = AnomalySuppressor(
            self.model, layer_idx=self.layer_idx, q=self.q, kappa=self.kappa
        )
        suppressor.install()
        try:
            prompt_embeds, prompt_mask, negative_embeds, negative_mask = (
                self.model.encode_prompt(caption)
            )

            def step_callback(step, timestep, latents):
                suppressor.set_step(step + 1)

            latents = self.model(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_attention_mask=negative_mask,
                num_images_per_prompt=1,
                num_inference_steps=self.num_inf_steps,
                callback=step_callback,
                callback_steps=1,
                output_type="latent",
                generator=self.generator,
            ).images
        finally:
            suppressor.remove()

        decoded = self.model.vae.decode(
            latents / self.model.vae.config.scaling_factor, return_dict=False
        )[0]
        return self.model.image_processor.postprocess(decoded, output_type="pil")[0]
