from .nodes import SamplerInversedEulerNode, MixNoiseNode, CombineNoiseLatentNode


NODE_CLASS_MAPPINGS = {
    "SamplerInversedEulerNode": SamplerInversedEulerNode,
    "MixNoiseNode": MixNoiseNode,
    "CombineNoiseLatentNode": CombineNoiseLatentNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerInversedEulerNode": "Inversed Euler Sampler",
    "MixNoiseNode": "Mix Noise with Latent",
    "CombineNoiseLatentNode": "Combine Latent Noise",
}
