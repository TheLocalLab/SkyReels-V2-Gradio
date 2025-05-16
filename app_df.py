import os
import gc
import time
import random
import torch
import imageio
import gradio as gr
from diffusers.utils import load_image

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer, resizecrop

#is_shared_ui = True if "fffiloni/SkyReels-V2" in os.environ['SPACE_ID'] else False
is_shared_ui = False

model_id = None
if is_shared_ui:
    model_id = download_model("Skywork/SkyReels-V2-DF-1.3B-540P")

def generate_diffusion_forced_video(
    prompt,
    image=None,
    target_length="10",
    model_id="Skywork/SkyReels-V2-DF-1.3B-540P",
    resolution="540P",
    num_frames=257,
    ar_step=0,
    causal_attention=False,
    causal_block_size=1,
    base_num_frames=97,
    overlap_history=17,
    addnoise_condition=20,
    guidance_scale=6.0,
    shift=8.0,
    inference_steps=30,
    use_usp=False,
    offload=True,
    fps=24,
    seed=None,
    prompt_enhancer=False,
    teacache=True,
    teacache_thresh=0.2,
    use_ret_steps=True,
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

    if num_frames > base_num_frames and overlap_history is None:
        raise ValueError("Specify `overlap_history` for long video generation. Try 17 or 37.")
    if addnoise_condition > 60:
        print("Warning: Large `addnoise_condition` may reduce consistency. Recommended: 20.")

    if image is not None:
        image = load_image(image).convert("RGB")
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        image = resizecrop(image, height, width)

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    prompt_input = prompt
    if prompt_enhancer and image is None:
        enhancer = PromptEnhancer()
        prompt_input = enhancer(prompt_input)
        del enhancer
        gc.collect()
        torch.cuda.empty_cache()

    pipe = DiffusionForcingPipeline(
        model_id,
        dit_path=model_id,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=use_usp,
        offload=offload,
    )

    if causal_attention:
        pipe.transformer.set_ar_attention(causal_block_size)

    if teacache:
        if ar_step > 0:
            num_steps = (
                inference_steps + (((base_num_frames - 1) // 4 + 1) // causal_block_size - 1) * ar_step
            )
        else:
            num_steps = inference_steps
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=num_steps,
            teacache_thresh=teacache_thresh,
            use_ret_steps=use_ret_steps,
            ckpt_dir=model_id,
        )

    with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=inference_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            overlap_history=overlap_history,
            addnoise_condition=addnoise_condition,
            base_num_frames=base_num_frames,
            ar_step=ar_step,
            causal_block_size=causal_block_size,
            fps=fps,
        )[0]

    os.makedirs("gradio_df_videos", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"gradio_df_videos/{prompt[:50].replace('/', '')}_{seed}_{timestamp}.mp4"
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    return output_path


# Gradio UI
resolution_options = ["540P", "720P"]
model_options = ["Skywork/SkyReels-V2-DF-1.3B-540P"]  # Update if there are more

if is_shared_ui is False:
    model_options = [
        "Skywork/SkyReels-V2-DF-1.3B-540P",
        "Skywork/SkyReels-V2-DF-14B-540P",
        "Skywork/SkyReels-V2-DF-14B-720P"
    ]

length_options = []
if is_shared_ui is True:
    length_options = ["4", "10"]
else:
    length_options = ["4", "10", "15", "30", "60"]

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# SkyReels V2: Infinite-Length Film Generation")
        gr.Markdown("The first open-source video generative model employing AutoRegressive Diffusion-Forcing architecture that achieves the SOTA performance among publicly available models.")

        gr.HTML("""
            <div style="display:flex;column-gap:4px;">
                <a href="https://github.com/SkyworkAI/SkyReels-V2">
                    <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
                </a> 
    			<a href="https://arxiv.org/pdf/2504.13074">
                    <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
                </a>
                <a href="https://huggingface.co/spaces/fffiloni/SkyReels-V2?duplicate=true">
    				<img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
    			</a>	
            </div>
        """)
        with gr.Row():

            with gr.Column():

                prompt = gr.Textbox(label="Prompt")

                with gr.Row():
                    if is_shared_ui:
                        target_length = gr.Radio(label="Video length target", choices=length_options, value="4")
                        forbidden_length = gr.Radio(label="Available target on duplicated instance", choices=["15","30","60"], value=None, interactive=False)
                    else:
                        target_length = gr.Radio(label="Video length target", choices=length_options, value="4")
                
                num_frames = gr.Slider(minimum=17, maximum=257, value=97, step=20, label="Number of Frames", interactive=False)
                image = gr.Image(type="filepath", label="Input Image (optional)")
                
                with gr.Accordion("Advanced Settings", open=False):
                    model_id = gr.Dropdown(choices=model_options, value=model_options[0], label="Model ID")
                    resolution = gr.Radio(choices=resolution_options, value="540P", label="Resolution", interactive=False if is_shared_ui else True)
                    ar_step = gr.Number(label="AR Step", value=0)
                    causal_attention = gr.Checkbox(label="Causal Attention")
                    causal_block_size = gr.Number(label="Causal Block Size", value=1)
                    base_num_frames = gr.Number(label="Base Num Frames", value=97)
                    overlap_history = gr.Number(label="Overlap History (set for long videos)", value=None)
                    addnoise_condition = gr.Number(label="AddNoise Condition", value=0)
                    guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=6.0, step=0.1, label="Guidance Scale")
                    shift = gr.Slider(minimum=0.0, maximum=20.0, value=8.0, step=0.1, label="Shift")
                    inference_steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Inference Steps")
                    use_usp = gr.Checkbox(label="Use USP", visible=False if is_shared_ui else True)
                    offload = gr.Checkbox(label="Offload", value=True, interactive=False if is_shared_ui else True)
                    fps = gr.Slider(minimum=1, maximum=60, value=24, step=1, label="FPS")
                    seed = gr.Number(label="Seed (optional)", precision=0)
                    prompt_enhancer = gr.Checkbox(label="Prompt Enhancer", visible=False if is_shared_ui else True)
                    use_teacache = gr.Checkbox(label="Use TeaCache", value=True)
                    teacache_thresh = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.01, label="TeaCache Threshold")
                    use_ret_steps = gr.Checkbox(label="Use Retention Steps", value=True)

                submit_btn = gr.Button("Generate")

            with gr.Column():

                output_video = gr.Video(label="Generated Video")

                gr.Examples(
                    examples = [
                        ["A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface, with the swan occasionally dipping its head into the water to feed.", "./examples/swan.jpeg", "10"],
                       # ["A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface, with the swan occasionally dipping its head into the water to feed.", None],
                        ["A sea turtle swimming near a shipwreck", "./examples/turtle.jpeg", "10"],
                       # ["A sea turtle swimming near a shipwreck", None],
                    ],
                    fn = generate_diffusion_forced_video,
                    inputs = [prompt, image, target_length],
                    outputs = [output_video],
                    cache_examples = True,
                    cache_mode = "lazy"
                )

    def set_num_frames(target_l):

        n_frames = 0
        overlap_history = 0
        addnoise_condition = 0
        ar_step = 0
        causal_attention = False
        causal_block_size = 1
        use_teacache = True
        teacache_thresh = 0.2
        use_ret_steps = True

        if target_l == "4":
            n_frames = 97
            use_teacache = True
            teacache_thresh = 0.2
            use_ret_steps = True
        elif target_l == "10":
            n_frames = 257
            overlap_history = 17
            addnoise_condition = 20
            use_teacache = True
            teacache_thresh = 0.2
            use_ret_steps = True
        elif target_l == "15":
            n_frames = 377
            overlap_history = 17
            addnoise_condition = 20
            use_teacache = True
            teacache_thresh = 0.3
            use_ret_steps = True
        elif target_l == "30":
            n_frames = 737
            overlap_history = 17
            addnoise_condition = 20
            use_teacache = True
            teacache_thresh = 0.3
            use_ret_steps = True
            causal_attention = False
            ar_step = 0
            causal_block_size = 1
        elif target_l == "60":
            n_frames = 1457
            overlap_history = 17
            addnoise_condition = 20
            use_teacache = True
            teacache_thresh = 0.3
            use_ret_steps = True
            causal_attention = False
            ar_step = 0
            causal_block_size = 0
        
        return n_frames, overlap_history, addnoise_condition, ar_step, causal_attention, causal_block_size, use_teacache, teacache_thresh, use_ret_steps
        

    target_length.change(
        fn = set_num_frames,
        inputs = [target_length],
        outputs = [num_frames, overlap_history, addnoise_condition, ar_step, causal_attention, causal_block_size, use_teacache, teacache_thresh, use_ret_steps],
        queue = False
    )

    submit_btn.click(
        fn = generate_diffusion_forced_video,
        inputs = [
            prompt,
            image,
            target_length,
            model_id,
            resolution,
            num_frames,
            ar_step,
            causal_attention,
            causal_block_size,
            base_num_frames,
            overlap_history,
            addnoise_condition,
            guidance_scale,
            shift,
            inference_steps,
            use_usp,
            offload,
            fps,
            seed,
            prompt_enhancer,
            use_teacache,
            teacache_thresh,
            use_ret_steps
        ],
        outputs = [
            output_video
        ]
    )

demo.launch(show_error=True, show_api=False, share=False)
