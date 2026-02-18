import math
from typing import Optional

import torch
from torch import Tensor


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Tensor] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
    ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._betas_for_alpha_bar(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        self.final_alpha_cumprod = torch.tensor(1.0)

    @staticmethod
    def _betas_for_alpha_bar(num_diffusion_timesteps: int) -> Tensor:
        def alpha_bar(time_step: float) -> float:
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas)

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()
        self.timesteps = self.timesteps.long().to(device)

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
    ) -> Tensor:
        t = timestep
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps

        if (
            model_output.shape[1] == sample.shape[1] * 2
            and self.variance_type == "fixed_small"
        ):
            model_output, _ = torch.split(model_output, sample.shape[1], dim=1)

        if self.prediction_type == "epsilon":
            pred_original_sample = self._get_variance(sample, model_output, t)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                model_output * (-self.sqrt_recip_alphas_cumprod[t])
                + torch.sqrt(1 - self.alphas_cumprod[t]) * sample
            )
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        predicted_variance = self._get_variance(model_output, sample, t)
        beta_prod_t = 1 - self.alphas_cumprod[t]
        beta_prod_t_prev = (
            1 - self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )

        pred_sample_direction = (
            torch.sqrt(beta_prod_t_prev)
            * s_noise
            * torch.sqrt(1 - beta_prod_t)
            / torch.sqrt(1 - self.alphas_cumprod[t])
            * (
                model_output
                - torch.sqrt(1 - self.alphas_cumprod[t])
                / torch.sqrt(beta_prod_t)
                * pred_original_sample
            )
        )

        prev_sample = (
            torch.sqrt(beta_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        if self.variance_type == "fixed_small":
            variance = self._get_variance(model_output, sample, t)
            variance = torch.clamp(variance, min=1e-20)
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + torch.sqrt(variance) * noise
        elif self.variance_type == "fixed_large":
            variance = self.betas[t]
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + torch.sqrt(variance) * noise
        elif self.variance_type == "scaled_large":
            variance = self.betas[t]
            prev_sample = prev_sample + torch.sqrt(
                variance
            ) * s_noise * torch.randn_like(sample)
        elif self.variance_type == "fixed_small_log":
            variance = self._get_variance(model_output, sample, t)
            variance = torch.log(torch.clamp(variance, min=1e-20))
            variance = torch.exp(0.5 * variance)
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + noise * variance
        elif self.variance_type == "learned":
            variance = model_output[:, sample.shape[1] :]
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + variance * noise
        elif self.variance_type == "learned_range":
            v = model_output[:, sample.shape[1] :]
            sigma_min = self.betas[prev_t] ** 0.5 if prev_t >= 0 else 0
            sigma_max = self.betas[t] ** 0.5
            c_skip = sigma_max**2 / (sigma_max**2 + sigma_min**2)
            c_out = sigma_min * sigma_max / torch.sqrt(sigma_max**2 - sigma_min**2)
            prev_sample = c_skip * sample + c_out * (
                v * (sample - pred_original_sample) + pred_original_sample
            )

        return prev_sample

    def _get_variance(self, model_output: Tensor, sample: Tensor, t: int) -> Tensor:
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[t - 1] if t > 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.prediction_type == "epsilon":
            beta_prod_t_prev = beta_prod_t_prev * (
                (1 - alpha_prod_t) / (1 - alpha_prod_t)
            )
            variance = (
                beta_prod_t_prev
                * (alpha_prod_t_prev / alpha_prod_t) ** 2
                * (1 - alpha_prod_t / alpha_prod_t_prev)
            )
        elif self.prediction_type == "sample":
            variance = beta_prod_t_prev
        elif self.prediction_type == "v_prediction":
            variance = beta_prod_t
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return variance

    def add_noise(
        self, original_samples: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples


class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Tensor] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = False,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
    ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.steps_offset = steps_offset

    def set_timesteps(self, num_inference_steps: int, offset: int = 0):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()
        self.timesteps = self.timesteps.long()
        self.timesteps += offset

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        eta: float = 0.0,
    ) -> Tensor:
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**0.5

        pred_sample_direction = (
            1 - alpha_prod_t_prev - std_dev_t**2
        ) ** 0.5 * model_output

        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            noise = torch.randn_like(sample)
            variance_noise = std_dev_t * noise
            prev_sample = prev_sample + variance_noise

        return prev_sample

    def _get_variance(self, timestep: int, prev_timestep: int) -> Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance

    def add_noise(
        self, original_samples: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples


class DPMSolverMultistepScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Tensor] = None,
        algorithm_type: str = "dpmsolver++",
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
    ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.algorithm_type = algorithm_type
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.lower_order_final = lower_order_final
        self.euler_at_final = euler_at_final

        self.num_inference_steps = None
        self.timesteps = None
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()
        self.timesteps = self.timesteps.long().to(device)
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
    ) -> Tensor:
        if self.algorithm_type == "dpmsolver++":
            return self._step_dpmsolver_pp(model_output, timestep, sample)
        elif self.algorithm_type == "dpmsolver":
            return self._step_dpmsolver(model_output, timestep, sample)
        else:
            raise ValueError(f"Unknown algorithm_type: {self.algorithm_type}")

    def _step_dpmsolver_pp(
        self, model_output: Tensor, timestep: int, sample: Tensor
    ) -> Tensor:
        step_index = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        t = timestep
        t_next = (
            self.timesteps[step_index + 1]
            if step_index + 1 < len(self.timesteps)
            else 0
        )

        mtp_num = self.solver_order

        self.model_outputs.append(model_output)
        self.model_outputs.pop(0)

        if self.lower_order_final and step_index + 2 > self.num_inference_steps // 2:
            mtp_num = 1
        elif self.lower_order_nums < self.solver_order:
            mtp_num = self.lower_order_nums

        timestep_list = self.timesteps[
            step_index - mtp_num + 1 : step_index + 1
        ].tolist()
        model_output_list = self.model_outputs[-mtp_num:]

        if self.euler_at_final and step_index + 2 == self.num_inference_steps:
            return self._euler_step(sample, model_output, t, t_next)

        if len(timestep_list) == 0:
            return sample

        return self._multistep_dpm_solver_pp_update(
            sample, model_output_list, timestep_list, t, t_next
        )

    def _multistep_dpm_solver_pp_update(
        self,
        sample: Tensor,
        model_output_list: list,
        timestep_list: list,
        t: int,
        t_next: int,
    ) -> Tensor:
        n = len(timestep_list)
        alpha_t = self.alphas_cumprod[t] ** 0.5
        alpha_t_next = (
            self.alphas_cumprod[t_next] ** 0.5 if t_next >= 0 else torch.tensor(1.0)
        )
        beta_t = 1 - alpha_t / alpha_t_next

        if len(model_output_list) == 0:
            return sample

        if len(model_output_list) == 1:
            predicted_noise = model_output_list[0]
            if self.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_t**0.5 * predicted_noise) / (
                    1 - beta_t
                ) ** 0.5
            elif self.prediction_type == "v_prediction":
                pred_original_sample = alpha_t * sample - beta_t**0.5 * predicted_noise
            else:
                pred_original_sample = predicted_noise
            sample = (alpha_t_next / alpha_t) * sample - (
                alpha_t_next / alpha_t
            ) * beta_t**0.5 * predicted_noise
            return sample

        lambda_t = torch.log(alpha_t) - torch.log(alpha_t_next)
        lambda_s_list = [
            torch.log(self.alphas_cumprod[timestep])
            - torch.log(
                self.alphas_cumprod[timestep + 1]
                if timestep + 1 < self.num_train_timesteps
                else self.alphas_cumprod[0]
            )
            for timestep in timestep_list
        ]

        h = lambda_t
        r1 = h / lambda_s_list[-1]
        r2 = (
            h / (lambda_s_list[-1] - lambda_s_list[-2]) if len(lambda_s_list) > 1 else 0
        )

        D1 = model_output_list[-1]
        if len(model_output_list) > 1:
            D2 = (model_output_list[-1] - model_output_list[-2]) / (
                lambda_s_list[-1] - lambda_s_list[-2]
            )
        else:
            D2 = 0

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - (1 - alpha_t**2) ** 0.5 * D1) / alpha_t
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_t * sample - (1 - alpha_t**2) ** 0.5 * D1
        else:
            pred_original_sample = D1

        if self.solver_order == 2:
            if len(model_output_list) > 2:
                D3 = (model_output_list[-2] - model_output_list[-3]) / (
                    lambda_s_list[-2] - lambda_s_list[-3]
                )
            else:
                D3 = 0

            tor_sqrth = torch.sqrt(1 - torch.exp(-2 * r2 * lambda_s_list[-1]))
            h_ = 0.5 * r2 * h * h + (1 + r1 / 2) * (
                lambda_s_list[-1] - lambda_s_list[-2]
            )
            sample = alpha_t_next * (
                sample / alpha_t + h_ * (D1 + r2 * D2 + r1 * r2 * D3)
            ) - (1 - torch.exp(-2 * r2 * lambda_s_list[-1])) ** 0.5 * (
                h_ * (1 + r1) * D1 + r1 * (h_ - r1) * D2 + r1 * r2 * (h_ - r1) * D3
            )

        return sample

    def _step_dpmsolver(
        self, model_output: Tensor, timestep: int, sample: Tensor
    ) -> Tensor:
        step_index = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        t = timestep
        t_next = (
            self.timesteps[step_index + 1]
            if step_index + 1 < len(self.timesteps)
            else 0
        )

        self.model_outputs.append(model_output)
        self.model_outputs.pop(0)

        if self.lower_order_nums < self.solver_order:
            return self._euler_step(sample, model_output, t, t_next)

        return self._multistep_dpm_solver_update(
            sample,
            self.model_outputs,
            self.timesteps[step_index - 1 : step_index + 1].tolist(),
            t,
            t_next,
        )

    def _multistep_dpm_solver_update(
        self,
        sample: Tensor,
        model_output_list: list,
        timestep_list: list,
        t: int,
        t_next: int,
    ) -> Tensor:
        n = len(timestep_list)
        alpha_t = self.alphas_cumprod[t] ** 0.5
        alpha_t_next = (
            self.alphas_cumprod[t_next] ** 0.5 if t_next >= 0 else torch.tensor(1.0)
        )
        beta_t = 1 - alpha_t / alpha_t_next

        if self.prediction_type == "epsilon":
            predicted_noise = model_output_list[-1]
            pred_original_sample = (sample - beta_t**0.5 * predicted_noise) / (
                1 - beta_t
            ) ** 0.5
        elif self.prediction_type == "v_prediction":
            predicted_noise = model_output_list[-1]
            pred_original_sample = alpha_t * sample - beta_t**0.5 * predicted_noise
        else:
            pred_original_sample = model_output_list[-1]

        if n == 1:
            return self._euler_step(sample, predicted_noise, t, t_next)

        lambda_t = torch.log(alpha_t) - torch.log(alpha_t_next)
        lambda_s_list = [
            torch.log(self.alphas_cumprod[timestep])
            - torch.log(
                self.alphas_cumprod[timestep + 1]
                if timestep + 1 < self.num_train_timesteps
                else self.alphas_cumprod[0]
            )
            for timestep in timestep_list[:-1]
        ]

        h = lambda_t

        if self.solver_order == 2:
            r1 = h / lambda_s_list[-1]
            r2 = h / (lambda_s_list[-1] - lambda_s_list[-2])

            D1 = model_output_list[-1]
            D2 = (model_output_list[-1] - model_output_list[-2]) / (
                lambda_s_list[-1] - lambda_s_list[-2]
            )

            sample = (
                sample
                - self._sigma_bar(lambda_t) * (r1 + r2) * D1
                + self._sigma_bar(lambda_t) * r2 * D2
            )

        return sample

    def _sigma_bar(self, lambda_t: Tensor) -> Tensor:
        return torch.expm1(lambda_t)

    def _euler_step(
        self, sample: Tensor, model_output: Tensor, t: int, t_next: int
    ) -> Tensor:
        alpha_prod_t = self.alphas_cumprod[t] ** 0.5
        alpha_prod_t_next = (
            self.alphas_cumprod[t_next] ** 0.5 if t_next >= 0 else torch.tensor(1.0)
        )
        beta_prod_t = 1 - self.alphas_cumprod[t]
        beta_prod_t_next = (
            1 - self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(0.0)
        )

        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t
            dt = beta_prod_t**0.5 - beta_prod_t_next**0.5
            sample = sample - pred_original_sample * dt
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                alpha_prod_t * sample - beta_prod_t**0.5 * model_output
            )
            dt = beta_prod_t**0.5 - beta_prod_t_next**0.5
            sample = sample - pred_original_sample * dt
        else:
            sample = sample - model_output * (beta_prod_t**0.5 - beta_prod_t_next**0.5)

        return sample


class EulerDiscreteScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Tensor] = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: bool = False,
    ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.prediction_type = prediction_type
        self.interpolation_type = interpolation_type
        self.use_karras_sigmas = use_karras_sigmas

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        self.num_inference_steps = num_inference_steps

        if self.use_karras_sigmas:
            self.timesteps = self._get_karras_timesteps(num_inference_steps)
        else:
            step_ratio = self.num_train_timesteps // num_inference_steps
            self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()
            self.timesteps = self.timesteps.long().to(device)
            self.timesteps = torch.cat(
                [self.timesteps, torch.zeros_like(self.timesteps[:1])]
            ).to(device)

        self.sigmas = self.sigmas.to(device)
        self._index = 0

    def _get_karras_timesteps(self, num_inference_steps: int) -> Tensor:
        sigmas_min = float(self.sigmas[-1])
        sigmas_max = float(self.sigmas[0])

        step_ratio = (sigmas_max - sigmas_min) / (num_inference_steps - 1)
        sigmas = sigmas_max - step_ratio * torch.arange(num_inference_steps)
        sigmas = torch.cat([sigmas, sigmas_min * torch.ones(1)])
        return (
            ((sigmas / (1 + sigmas**2)) ** 0.5 * self.num_train_timesteps)
            .round()
            .long()
        )

    def step(
        self,
        model_output: Tensor,
        sigma: float,
        sample: Tensor,
    ) -> Tensor:
        if self._index >= len(self.timesteps) - 1:
            return sample

        t = self.timesteps[self._index].item()
        t_next = self.timesteps[self._index + 1].item()

        sigma_from = self.sigmas[self._index]
        sigma_to = self.sigmas[self._index + 1]

        if self.interpolation_type == "linear":
            sigma_inter = (
                sigma_from - (sigma_from - sigma_to) * (t / t_next)
                if t_next != 0
                else sigma_to
            )
        else:
            sigma_inter = sigma_from ** (1 - t / t_next) * sigma_to ** (t / t_next)

        if self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_inter * model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                sample * self.sqrt_alphas_cumprod[t] - sigma_inter * model_output
            )
        else:
            pred_original_sample = model_output

        sigma_up = self._get_variance(t, t_next) ** 0.5
        sigma_down = (sigma_up**2 / (sigma_inter**2 + sigma_up**2)) ** 0.5
        sigma_intermediate = (sigma_inter**2 - sigma_down**2) ** 0.5

        prev_sample = (
            pred_original_sample + (sigma_down - sigma_intermediate) * model_output
        )

        noise = torch.randn_like(sample)
        prev_sample = prev_sample + noise * sigma_up

        self._index += 1
        return prev_sample

    def _get_variance(self, t: int, t_next: int) -> float:
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_next = (
            self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        variance = (beta_prod_t_next / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_next
        )
        return variance.item()


class LMSDiscreteScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Tensor] = None,
        prediction_type: str = "epsilon",
    ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.prediction_type = prediction_type
        self.derivatives = []

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu"):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()
        self.timesteps = self.timesteps.long().to(device)
        self.derivatives = []

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        order: int = 4,
    ) -> Tensor:
        step_index = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()

        sigma = (
            self.sqrt_one_minus_alphas_cumprod[timestep]
            / self.sqrt_alphas_cumprod[timestep]
        )

        if self.prediction_type == "epsilon":
            derivative = (sample - model_output * sigma) / sigma
        elif self.prediction_type == "v_prediction":
            derivative = (
                self.sqrt_alphas_cumprod[timestep] * model_output
                - self.sqrt_one_minus_alphas_cumprod[timestep] * sample
            )
        else:
            derivative = model_output

        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)

        t = self.timesteps[step_index]
        t_next = (
            self.timesteps[step_index + 1]
            if step_index + 1 < len(self.timesteps)
            else 0
        )

        dt = t - t_next
        step_size = dt / order

        sample = sample - self._lms_coefficients(order, step_size)

        return sample

    def _lms_coefficients(self, order: int, step_size: float) -> Tensor:
        coeffs = [1.0]
        for i in range(1, order):
            coeff = step_size * sum(
                (step_size**j) / math.factorial(j + 1) * coeffs[i - 1 - j]
                for j in range(i)
            )
            coeffs.append(coeff)

        derivative_sum = sum(
            coeff * deriv for coeff, deriv in zip(coeffs, reversed(self.derivatives))
        )

        return step_size * derivative_sum
