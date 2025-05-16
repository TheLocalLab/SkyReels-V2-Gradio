import os
import gc
import time
import random
import imageio
import torch
import gradio as gr
from diffusers.utils import load_image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline, Text2VideoPipeline, PromptEnhancer, resizecrop

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}

def generate_video(
    prompt,
    model_id,
    resolution,
    num_frames,
    image=None,
    guidance_scale=6.0,
    shift=8.0,
    inference_steps=30,
    use_usp=False,
    offload=False,
    fps=24,
    seed=None,
    prompt_enhancer=False,
    teacache=False,
    teacache_thresh=0.2,
    use_ret_steps=False
):
    model_id = download_model(model_id)
    
    if resolution == "540P":
        height, width = 544, 960
    elif resolution == "720P":
        height, width = 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {resolution}")

    if seed is None:
        random.seed(time.time())
        seed = int(random.randrange(4294967294))
    
    if image is not None:
        image = load_image(image).convert("RGB")
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        image = resizecrop(image, height, width)

    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
        "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
    )

    prompt_input = prompt
    if prompt_enhancer and image is None:
        enhancer = PromptEnhancer()
        prompt_input = enhancer(prompt_input)
        del enhancer
        gc.collect()
        torch.cuda.empty_cache()

    if image is None:
        pipe = Text2VideoPipeline(model_path=model_id, dit_path=model_id, use_usp=use_usp, offload=offload)
    else:
        pipe = Image2VideoPipeline(model_path=model_id, dit_path=model_id, use_usp=use_usp, offload=offload)

    if teacache:
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=inference_steps,
            teacache_thresh=teacache_thresh,
            use_ret_steps=use_ret_steps,
            ckpt_dir=model_id,
        )

    kwargs = {
        "prompt": prompt_input,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "num_inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "height": height,
        "width": width,
    }

    if image is not None:
        kwargs["image"] = image.convert("RGB")

    with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(**kwargs)[0]

    os.makedirs("gradio_videos", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"gradio_videos/{prompt[:50].replace('/', '')}_{seed}_{timestamp}.mp4"
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    
    return output_path

# Gradio UI
resolution_options = ["540P", "720P"]
model_options = MODEL_ID_CONFIG["text2video"] + MODEL_ID_CONFIG["image2video"]

app = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Dropdown(choices=model_options, value="Skywork/SkyReels-V2-I2V-1.3B-540P", label="Model ID", interactive=False),
        gr.Radio(choices=resolution_options, value="540P", label="Resolution", interactive=False),
        gr.Slider(minimum=16, maximum=200, value=97, step=1, label="Number of Frames"),
        gr.Image(type="filepath", label="Input Image (optional)"),
        gr.Slider(minimum=1.0, maximum=20.0, value=5.0, step=0.1, label="Guidance Scale"),
        gr.Slider(minimum=0.0, maximum=20.0, value=3.0, step=0.1, label="Shift"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Inference Steps"),
        gr.Checkbox(label="Use USP"),
        gr.Checkbox(label="Offload", value=True, interactive=False),
        gr.Slider(minimum=1, maximum=60, value=24, step=1, label="FPS"),
        gr.Number(label="Seed (optional, random if empty)", precision=0),
        gr.Checkbox(label="Prompt Enhancer"),
        gr.Checkbox(label="Use TeaCache"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.01, label="TeaCache Threshold"),
        gr.Checkbox(label="Use Retention Steps"),
    ],
    outputs=gr.Video(label="Generated Video"),
    title="SkyReels V2 Video Generator",
)

app.launch(show_api=False, show_error=True, ssr_mode=False)
