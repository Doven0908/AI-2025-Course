import os
from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *

# 文件夹路径
background_folder = 'E:/AppData/PythonProjects/Daily_Practice/Crack_generation/task_1/base'

# 获取背景和掩码图片文件列表
background_images = [f for f in os.listdir(background_folder) if f.endswith('.jpg') or f.endswith('.png')]

with Workflow():
    model, clip, vae = CheckpointLoaderSimple(r'sdxl\万享XL_超写实摄影V8.4_V8.4.safetensors')
    _, prompt_strings = EasyPromptList('((masterpiece)), ((best quality)), "dramatic thunderstorm, powerful lightning bolts, electrified sky, torrential rain, atmospheric tempest"', '((masterpiece)), ((best quality)), "The river flows through the forest"', '((masterpiece)), ((best quality)), "realistic, smooth concrete surface, restored, no visible cracks, seamless texture, The crack has been repaired, concrete surface repair, no visible damage, smooth and clean"', '', '', None)
    conditioning = CLIPTextEncode(prompt_strings, clip)
    conditioning2 = CLIPTextEncode('"blurry, distorted repair, visible artifacts, inconsistent texture, crack"', clip)
    control_net = ControlNetLoader('diffusion_pytorch_model_promax.safetensors')
    for background_file in background_images:
        background_path = os.path.join(background_folder, background_file)
        image, _ = LoadImage(background_path)
        image2 = LineArtPreprocessor(image, 'disable', 512)
        mask = ImageToMask(image2, 'red')
        control_netpositive, control_netnegative = ControlNetInpaintingAliMamaApply(conditioning, conditioning2, control_net, vae, image, mask, None, None, None)
        latent = VAEEncode(image, vae)
        latent = RepeatLatentBatch(latent, 3)
        latent = KSampler(model, 114514, 30, 5, 'dpmpp_2s_ancestral_cfg_pp', 'karras', control_netpositive, control_netnegative, latent, 1)
        image3 = VAEDecode(latent, vae)
        save_name = f"base{background_file.split('.')[0]}"
        SaveImage(image3, f'StyleChange/{save_name}')