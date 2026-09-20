import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_PIPE = None


def load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        # xFormers is optional; the app still works without it.
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    _PIPE = pipe
    return pipe


def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt 不能为空")

    pipe = load_pipeline()
    seed = int(seed)
    if seed < 0:
        seed = int(time.time()) % 2_147_483_647

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip() or None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            generator=generator,
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{seed}.png"
    save_path = OUTPUT_DIR / filename
    result.images[0].save(save_path)

    info = (
        f"Model: {MODEL_ID}\nDevice: {DEVICE}\nSeed: {seed}\n"
        f"Steps: {int(steps)}\nCFG: {float(guidance_scale):g}\n"
        f"Size: {int(width)}x{int(height)}\nSaved: {save_path}"
    )
    return result.images[0], info


def build_demo():
    with gr.Blocks(title="Local Image Generation") as demo:
        gr.Markdown("# Local Image Generation\nStable Diffusion 1.5 · 8GB VRAM preset")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    value="a futuristic photonic AI chip laboratory, cinematic lighting, highly detailed",
                )
                negative = gr.Textbox(
                    label="Negative prompt",
                    lines=3,
                    value="blurry, low quality, distorted, bad anatomy, extra fingers",
                )
                with gr.Row():
                    steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
                    cfg = gr.Slider(1, 12, value=7, step=0.5, label="CFG")
                with gr.Row():
                    width = gr.Dropdown([384, 448, 512, 640], value=512, label="Width")
                    height = gr.Dropdown([384, 448, 512, 640], value=512, label="Height")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                button = gr.Button("Generate", variant="primary")
            with gr.Column():
                image = gr.Image(label="Result", type="pil")
                info = gr.Textbox(label="Generation info", lines=7)

        button.click(
            generate_image,
            [prompt, negative, steps, cfg, width, height, seed],
            [image, info],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860)
