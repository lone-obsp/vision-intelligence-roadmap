# Local image generation

A minimal Stable Diffusion 1.5 + Diffusers + Gradio app for Windows machines with about 8GB VRAM.

## Setup (PowerShell)

From this directory, activate the environment where CUDA-enabled PyTorch is already installed:

```powershell
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860 in your browser. The first run downloads the model and can take a while.

Recommended first settings:

- 512 x 512
- 20 steps
- CFG 7
- seed -1

Generated images are saved under `outputs/` (the directory is created automatically).

## Optional model override

Set `MODEL_ID` before starting the app:

```powershell
$env:MODEL_ID="runwayml/stable-diffusion-v1-5"
python app.py
```

Keep batch size at one. If CUDA out-of-memory occurs, close other GPU programs and use 448 x 448.

## Troubleshooting

- Check CUDA with `python -c "import torch; print(torch.cuda.is_available())"`.
- If xFormers installation fails, ignore it; it is optional.
- If Hugging Face asks for authentication, run `huggingface-cli login` or use a model that is publicly accessible.
