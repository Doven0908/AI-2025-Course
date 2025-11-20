from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *
with Workflow():
    model, clip, vae = CheckpointLoaderSimple('flux1-dev-fp8.safetensors')
    model, clip = LoraLoader(model, clip, 'flux_lora_QAZWSXroadcrack_rank16_bf16-step03000.safetensors', 0.5, 0.5)
    clip_text_encode_positive_prompt_conditioning = CLIPTextEncode('A highly detailed, close-up, macro photograph of ground QAZWSXroadcrack cracks on a weathered concrete surface.', clip)
    clip_text_encode_positive_prompt_conditioning = FluxGuidance(clip_text_encode_positive_prompt_conditioning, None)
    clip_text_encode_negative_prompt_conditioning = CLIPTextEncode('', clip)
    latent = EmptySD3LatentImage(1216, 832, 20)
    latent = KSampler(model, 114514, 20, 1, 'euler', 'simple', clip_text_encode_positive_prompt_conditioning, clip_text_encode_negative_prompt_conditioning, latent, 1)
    image = VAEDecode(latent, vae)
    SaveImage(image, f'LoraCrack/lora')