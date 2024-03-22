import torch
from tqdm import trange

from comfy.samplers import KSAMPLER


class SamplerInversedEulerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self):
        @torch.no_grad()
        def sample_inversed_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
            """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            for i in trange(1, len(sigmas), disable=disable):
                sigma_in = sigmas[i-1]

                if i == 1:
                    sigma_t = sigmas[i]
                else:
                    sigma_t = sigma_in

                denoised = model(x, sigma_t * s_in, **extra_args)

                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i-1]

                dt = sigmas[i] - sigmas[i-1]
                x = x + d * dt
                if callback is not None:
                    callback(
                        {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            return x

        ksampler = KSAMPLER(sample_inversed_euler)
        return (ksampler, )

# combined_noise = ((1 - randomness) * rec_noise + randomness * rand_noise) / ((randomness**2 + (1-randomness)**2) ** 0.5)


class MixNoiseNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent": ("LATENT", ),
            "randomness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "round": False}),
        }}

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "combine"

    def combine(self, latent, randomness):
        latent_image = latent['samples']
        rand_noise = torch.randn_like(latent_image)
        combined_noise = ((1 - randomness) * latent_image + randomness *
                          rand_noise) / ((randomness**2 + (1-randomness)**2) ** 0.5)
        latent = latent.copy()
        latent['samples'] = combined_noise
        return (latent, )
