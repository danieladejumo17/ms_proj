# ================= RUNNING COSMOS with Hugging Face
# git clone https://github.com/danieladejumo17/ms_proj.git
# cd my_cosmos-reason1
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env
# uv tool install -U "huggingface_hub[cli]"
# uv run scripts/inference_sample.py
# . .venv/bin/activate
# python -m ensurepip --upgrade
# python -m pip install --upgrade pip
# python -m pip install torchvision
# python -m pip install opencv-python
# python -m pip install bitsandbytes
# python scripts/fp8_test.py --video_dir ../stu_dataset/
# =================  END OF RUNNING COSMOS WITH HUGGINGFACE

# ================= RUNNING FP4 QUANTIZATION AND COSMOS WITH VLLM
pip install vllm llmcompressor torch torchvision transformers qwen_vl_utils opencv-python
pip install --upgrade compressed-tensors llmcompressor
pip uninstall -y torch torchvision vllm && pip install vllm torchvision
pip install open3d

# pip install --upgrade compressed-tensors llmcompressor transformers qwen_vl_utils opencv-python
# pip install vllm torchvision
# pip install datasets numpy decord
# ================= END OF RUNNING FP4 QUANTIZATION AND COSMOS WITH VLLM



# NOPENPENPENPEpip install --pre --upgrade torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# # 1. Uninstall the incompatible pre-built vLLM
# pip uninstall -y vllm

# # 2. Install build dependencies
# pip install --upgrade pip packaging ninja wheel setuptools

# # 3. Clone the vLLM repository
# git clone https://github.com/vllm-project/vllm.git
# cd vllm

# # 4. Configure environment for RTX 5090 (Blackwell)
# #    Force Flash Attention 2 (FA3 is not yet fully stable on Nightly+Blackwell contexts)
# export VLLM_FLASH_ATTN_VERSION=2
# #    Use the installed torch version for compilation
# export VLLM_USE_PRECOMPILED=0

# # 5. Compile and install (This will take ~10 mins)
# pip install -e .