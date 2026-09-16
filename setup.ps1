# One-time environment setup (run from the project folder in PowerShell)
$ErrorActionPreference = "Stop"
python -m venv .venv
# Avast intercepts HTTPS on this machine; point pip at its CA bundle so downloads verify.
@"
[global]
cert = C:\ProgramData\Avast Software\Avast\wscert.pem
"@ | Out-File -Encoding ascii .venv\pip.ini
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -c "import torch, ultralytics; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)"
