cd ms_proj/my_cosmos-reason1
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv tool install -U "huggingface_hub[cli]"
uv run scripts/inference_sample.py
. .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install torchvision
python -m pip install opencv-python
python -m pip install bitsandbytes
python scripts/fp8_test.py --video_dir ../stu_dataset/