import os
from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *

# 文件夹路径
background_folder = 'E:/AppData/PythonProjects/Daily_Practice/Crack_generation/task_4/base'
mask_folder = 'E:/AppData/PythonProjects/Daily_Practice/Crack_generation/task_4/mask'

# 获取背景和掩码图片文件列表
background_images = [f for f in os.listdir(background_folder) if f.endswith('.jpg') or f.endswith('.png')]
mask_images = [f for f in os.listdir(mask_folder) if f.endswith('.jpg') or f.endswith('.png')]

with Workflow():
    model = UNETLoader('flux1-dev-fp8.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'flux1-depth-dev-lora.safetensors', 1)
    model = LoraLoaderModelOnly(model, 'flux_lora_QAZWSXroadcrack_rank16_bf16-step03000.safetensors', 1)
    clip = DualCLIPLoader('clip_l.safetensors', 't5xxl_fp16.safetensors', 'flux', 'default')

    # 遍历所有背景和掩码图片
    for background_file in background_images:
        background_path = os.path.join(background_folder, background_file)
        image, _ = LoadImage(background_path)

        for mask_file in mask_images:
            mask_path = os.path.join(mask_folder, mask_file)
            mask_image, _ = LoadImage(mask_path)

            # 生成CLIP文本编码和其它处理步骤
            clip_text_encode_positive_prompt_conditioning = CLIPTextEncode(
                'A highly detailed, close-up, macro photograph of ground QAZWSXroadcrack cracks on a weathered concrete surface.',
                clip)
            clip_text_encode_positive_prompt_conditioning = FluxGuidance(clip_text_encode_positive_prompt_conditioning,
                                                                         None)
            clip_text_encode_negative_prompt_conditioning = CLIPTextEncode('', clip)

            vae = VAELoader('ae.safetensors')
            positive, negative, latent = InstructPixToPixConditioning(
                clip_text_encode_positive_prompt_conditioning,
                clip_text_encode_negative_prompt_conditioning,
                vae, image
            )

            control_net = ControlNetLoader('FLUX.1-dev-ControlNet-Union-Pro-2.0 .safetensors')

            control_netpositive, control_netnegative = ControlNetApplyAdvanced(
                positive, negative, control_net, mask_image, 1, 0, 1, vae
            )

            latent = RepeatLatentBatch(latent, 4)
            latent = KSampler(
                model, 114514, 20, 1, 'euler', 'normal', control_netpositive, control_netnegative, latent, 1
            )
            image5 = VAEDecode(latent, vae)

            # 保存合成图片
            save_name = f"base{background_file.split('.')[0]}_crack{mask_file.split('.')[0]}"
            SaveImage(image5, f'CrackSynthesis/{save_name}')

