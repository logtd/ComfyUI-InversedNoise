from .nodes import SamplerInversedEulerNode, MixNoiseNode


NODE_CLASS_MAPPINGS = {
    "SamplerInversedEulerNode": SamplerInversedEulerNode,
    "MixNoiseNode": MixNoiseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerInversedEulerNode": "Inversed Euler Sampler",
    "MixNoiseNode": "Mix Noise with Latent"
}
