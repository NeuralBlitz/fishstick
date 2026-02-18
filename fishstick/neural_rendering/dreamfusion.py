import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class ScoreDistillationSampling(nn.Module):
    def __init__(
        self,
        diffusion_model: nn.Module,
        guidance_scale: float = 7.5,
        noise_scheduler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.guidance_scale = guidance_scale
        self.noise_scheduler = noise_scheduler or DDPMNoiseScheduler()

    def compute_gradients(
        self,
        render_images: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = render_images.shape[0]

        if timestep is None:
            timestep = torch.randint(
                0,
                self.noise_scheduler.num_timesteps,
                (batch_size,),
                device=render_images.device,
            )

        noise = torch.randn_like(render_images)
        noisy_images = self.noise_scheduler.add_noise(render_images, noise, timestep)

        model_output = self.diffusion_model(noisy_images, timestep, prompt_embeddings)

        if self.guidance_scale > 1.0:
            unconditional_output = self.diffusion_model(
                noisy_images, timestep, torch.zeros_like(prompt_embeddings)
            )
            model_output = unconditional_output + self.guidance_scale * (
                model_output - unconditional_output
            )

        gradients = self._compute_image_gradient(
            model_output, render_images, noise, timestep
        )

        return gradients

    def _compute_image_gradient(
        self,
        model_output: torch.Tensor,
        original_images: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(model_output, noise, reduction="none")
        gradients = torch.autograd.grad(
            outputs=loss.sum(),
            inputs=original_images,
            create_graph=True,
            retain_graph=True,
        )[0]

        return gradients

    def forward(
        self,
        render_images: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        gradients = self.compute_gradients(render_images, prompt_embeddings)
        return gradients


class DDPMNoiseScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas", alphas)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        t = timestep
        prev_t = timestep - 1 if timestep > 0 else 0

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.ones_like(alpha_prod_t)
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        variance = (
            (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )
        variance = torch.clamp(variance, min=1e-20)

        pred_sample_direction = (1 - alpha_prod_t_prev - variance) ** 0.5 * model_output

        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        noise = (
            torch.randn_like(model_output)
            if timestep > 0
            else torch.zeros_like(model_output)
        )

        prev_sample = prev_sample + variance**0.5 * noise * 0

        return prev_sample


class ThreeDDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 4,
        time_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 12,
        attention_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim

        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.prompt_embedding = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.SiLU(),
        )

        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(TransformerBlock(hidden_dim, attention_heads))
        self.transformer_blocks = nn.ModuleList(layers)

        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        self.zero_out = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timestep.unsqueeze(-1).float() / self.num_timesteps)

        h = self.input_proj(x)

        if prompt_embeds is not None:
            prompt_emb = self.prompt_embedding(prompt_embeds)
            h = h + prompt_emb.unsqueeze(1)

        for block in self.transformer_blocks:
            h = block(h, t_emb)

        out = self.output_proj(h)
        out = self.zero_out(out)

        return out

    @property
    def num_timesteps(self) -> int:
        return 1000


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        x = (
            x
            + self.cross_attention(
                self.norm2(x),
                t_emb.unsqueeze(1).expand(-1, x.shape[1], -1),
                t_emb.unsqueeze(1).expand(-1, x.shape[1], -1),
            )[0]
        )

        x = x + self.mlp(self.norm2(x))

        return x


class DreamFusion(nn.Module):
    def __init__(
        self,
        nerf_model: nn.Module,
        diffusion_model: nn.Module,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        image_size: int = 64,
    ):
        super().__init__()
        self.nerf_model = nerf_model
        self.diffusion_model = diffusion_model
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.image_size = image_size

        self.sds_loss = ScoreDistillationSampling(
            diffusion_model=diffusion_model,
            guidance_scale=guidance_scale,
        )

        self.prompt_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

    def generate_camera_poses(
        self,
        num_poses: int,
        radius: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        angles = torch.linspace(0, 2 * math.pi, num_poses)

        poses = []
        for angle in angles:
            x = radius * torch.cos(angle)
            z = radius * torch.sin(angle)
            y = 0.0

            pose = torch.eye(4)
            pose[0, 3] = x
            pose[1, 3] = y
            pose[2, 3] = z

            pose[:3, :3] = self._look_at_rotation(
                torch.tensor([0.0, 0.0, 0.0], device=device),
                torch.tensor([x, y, z], device=device),
            )

            poses.append(pose)

        return torch.stack(poses)

    def _look_at_rotation(
        self,
        camera_position: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor = torch.tensor([0.0, 1.0, 0.0]),
    ) -> torch.Tensor:
        forward = F.normalize(target - camera_position, dim=0)
        right = F.normalize(torch.cross(forward, up), dim=0)
        up = torch.cross(right, forward)

        rotation = torch.eye(4)
        rotation[:3, 0] = right
        rotation[:3, 1] = up
        rotation[:3, 2] = -forward

        return rotation[:3, :3]

    def render_multiview(
        self,
        camera_poses: torch.Tensor,
        intrinsic: torch.Tensor,
        width: int = 64,
        height: int = 64,
    ) -> torch.Tensor:
        from .nerf import generate_camera_rays, ray_marching, volumetric_rendering

        device = camera_poses.device
        batch_size = camera_poses.shape[0]

        renderings = []

        for i in range(batch_size):
            camera_to_world = camera_poses[i : i + 1]

            rays_o, rays_d = generate_camera_rays(
                camera_to_world,
                intrinsic.unsqueeze(0),
                width,
                height,
            )

            rgb, sigma, t_vals = ray_marching(
                rays_o,
                rays_d,
                self.nerf_model,
                near=0.1,
                far=10.0,
                num_samples=64,
            )

            rendered_rgb, depth = volumetric_rendering(rgb, sigma, t_vals)
            renderings.append(rendered_rgb)

        return torch.stack(renderings)

    def score_distillation_loss(
        self,
        render_images: torch.Tensor,
        prompt: str,
    ) -> torch.Tensor:
        prompt_embeds = self.encode_prompt(prompt, render_images.device)

        gradients = self.sds_loss(render_images, prompt_embeds)

        loss = F.mse_loss(
            render_images, render_images.detach() - 1e-3 * gradients, reduction="mean"
        )

        return loss

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        num_variants: int = 1,
    ) -> torch.Tensor:
        dummy_embeds = torch.randn(1, 768, device=device)
        return dummy_embeds

    def forward(
        self,
        camera_poses: torch.Tensor,
        intrinsic: torch.Tensor,
        prompt: str,
    ) -> Dict[str, Any]:
        render_images = self.render_multiview(
            camera_poses,
            intrinsic,
            width=self.image_size,
            height=self.image_size,
        )

        sds_loss = self.score_distillation_loss(render_images, prompt)

        return {
            "rendered_images": render_images,
            "loss": sds_loss,
        }

    def optimize_nerf(
        self,
        prompt: str,
        num_iterations: int = 1000,
        camera_poses: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        lr: float = 5e-4,
    ) -> nn.Module:
        optimizer = torch.optim.Adam(self.nerf_model.parameters(), lr=lr)

        if camera_poses is None:
            camera_poses = self.generate_camera_poses(8, radius=4.0)

        if intrinsic is None:
            intrinsic = torch.tensor([[500, 0, 32], [0, 500, 32], [0, 0, 1]])

        self.nerf_model.train()

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            output = self.forward(camera_poses, intrinsic, prompt)
            loss = output["loss"]

            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}"
                )

        self.nerf_model.eval()

        return self.nerf_model


class MultiviewDiffusion(nn.Module):
    def __init__(
        self,
        base_diffusion: ThreeDDiffusionModel,
        num_views: int = 4,
    ):
        super().__init__()
        self.base_diffusion = base_diffusion
        self.num_views = num_views

        self.view_encoder = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.view_aggregator = nn.MultiheadAttention(128, 4, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        view_angles: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if view_angles is not None:
            view_features = self.view_encoder(view_angles)
            view_features = view_features.unsqueeze(1)

            aggregated, _ = self.view_aggregator(
                view_features, view_features, view_features
            )
            aggregated = aggregated.squeeze(1)

            x = x + aggregated.unsqueeze(1)

        return self.base_diffusion(x, timestep, prompt_embeds)

    def generate_views(
        self,
        num_views: int,
        latent_shape: Tuple[int, ...],
        prompt: str,
        device: torch.device,
    ) -> torch.Tensor:
        x = torch.randn(num_views, *latent_shape, device=device)

        timesteps = torch.linspace(
            self.base_diffusion.num_timesteps,
            0,
            self.base_diffusion.num_timesteps,
            device=device,
        ).long()

        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).expand(num_views)

            with torch.no_grad():
                noise_pred = self.forward(
                    x,
                    t_batch,
                    prompt_embeds=torch.randn(num_views, 768, device=device),
                )

            if i < len(timesteps) - 1:
                alpha_prod_t = self._get_alpha(t)
                alpha_prod_t_next = self._get_alpha(timesteps[i + 1])

                x = (x - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
                x = (
                    x * alpha_prod_t_next.sqrt()
                    + (1 - alpha_prod_t_next).sqrt() * torch.randn_like(x) * 0
                )

        return x

    def _get_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.linspace(0, 0.02, 1000, device=timestep.device)[timestep]


class StableDiffusion3DWrapper(nn.Module):
    def __init__(
        self,
        sd_model: nn.Module,
        image_size: int = 64,
    ):
        super().__init__()
        self.sd_model = sd_model
        self.image_size = image_size

        self.vae = VAEWrapper()

        self.upsampler = nn.Sequential(
            nn.Conv2d(4, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def render_to_image(self, render_output: torch.Tensor) -> torch.Tensor:
        if render_output.dim() == 4:
            render_output = render_output.unsqueeze(0)

        B, V, C, H, W = render_output.shape

        render_output = render_output.permute(0, 1, 3, 4, 2).reshape(B, V * H * W, C)

        latents = self.vae.encode(render_output)

        return latents

    def decode_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        images = self.vae.decode(latents)

        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return images

    def forward(
        self,
        nerf_render: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        latents = self.render_to_image(nerf_render)

        noise_pred = self.sd_model(latents, timestep, prompt_embeds)

        denoised = latents - noise_pred

        images = self.decode_to_image(denoised)

        return images


class VAEWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = x.reshape(B, V * C, H, W)

        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def three_D_diffusion(
    model: nn.Module,
    prompt: str,
    num_views: int = 4,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    device = next(model.parameters()).device

    latents = torch.randn(1, num_views, 4, 32, 32, device=device)

    timesteps = torch.linspace(1000, 0, 50, device=device).long()

    prompt_embeds = torch.randn(1, 768, device=device)

    for i, t in enumerate(timesteps):
        with torch.no_grad():
            noise_pred = model(latents, t.unsqueeze(0), prompt_embeds)

        if i < len(timesteps) - 1:
            alpha_prod_t = 1.0 - i / 1000.0
            alpha_prod_t_next = 1.0 - (i + 1) / 1000.0

            latents = (
                latents - (1 - alpha_prod_t).sqrt() * noise_pred
            ) / alpha_prod_t.sqrt()
            latents = (
                latents * alpha_prod_t_next.sqrt()
                + (1 - alpha_prod_t_next).sqrt() * torch.randn_like(latents) * 0
            )

    return latents
