# Environment setup (uv) — Speculative Decoding + Quantization homework

The homework needs **three isolated environments** because the speculators
training stack and the vLLM serving stack have conflicting dependencies. Each
script below uses [`uv`](https://docs.astral.sh/uv/) to create a Python 3.12
venv in the project root and install the pinned versions from the notebook.

> **Run these on the GPU box (Linux + NVIDIA H100 80GB), not on macOS.**
> `vllm==0.20.0`, `flash-attn`, and CUDA `torch` have no Apple-Silicon wheels,
> so the installs only succeed on a CUDA host.

## Environments

| Script | venv | Pinned packages | Purpose |
| --- | --- | --- | --- |
| `setup_speculators_venv.sh` | `speculators_venv` | `speculators` @ tag `v0.5.0` (editable) | Data prep, hidden-state generation, EAGLE-3 training |
| `setup_vllm_venv.sh` | `vllm_venv` | `vllm==0.20.0`, `fastapi<0.137` | Serving + `vllm bench serve` |
| `setup_comp_venv.sh` | `comp_venv` | `llmcompressor==0.12.0` | FP8 dynamic quantization |

## Usage

```bash
# from the project root, on the H100 box:
bash setup/setup_all.sh           # build all three, or run them individually:

bash setup/setup_speculators_venv.sh
bash setup/setup_vllm_venv.sh
bash setup/setup_comp_venv.sh
```

Activate one at a time:

```bash
source speculators_venv/bin/activate   # training
source vllm_venv/bin/activate          # serving / benchmarking
source comp_venv/bin/activate          # quantization
```

## Notes / troubleshooting

- **Do not submit the venvs.** If you init git here, add
  `speculators_venv/`, `vllm_venv/`, `comp_venv/`, and `speculators/` to
  `.gitignore`.
- **flash-attn build failures** (in `speculators_venv`): retry it after torch is
  present, without build isolation:
  `uv pip install --python speculators_venv/bin/python flash-attn --no-build-isolation`
- **`launch_vllm.py`**: the notebook refers to `scripts/launch_vllm.py` for
  starting the vLLM server — that script comes from the course/repo materials,
  not from these setup scripts. Drop it under `scripts/` before running the
  benchmark steps.
- **Versions** are pinned exactly as the notebook's "Required Library Versions"
  table specifies. If a pin is unavailable on your index, confirm the version
  string against the notebook before changing it.
- **Model / data**: `Qwen/Qwen3-8B` and ShareGPT-style conversations are pulled
  at runtime by the training/serving steps (e.g. via Hugging Face), not by these
  setup scripts.
