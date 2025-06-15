# plant-disease-identifier-pytorch
Help identify plant diseases - Focus on sugarcane and rice in indian context.

# Setup
```bash
# Assuming all the intel drivers are installed and working

# Install uv latest version
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize a new project (if not already done, creates pyproject.toml)
uv init

# Create a virtual environment (e.g., named.venv) using a specific Python version
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Installing Intel oneAPI Base Toolkit
sudo apt install intel-oneapi-base-toolkit
# Need to install using offline installer as the sudo apt is not working for some reason
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/4a5320d1-0b48-458d-9668-fd0e4501208c/intel-oneapi-base-toolkit-2025.1.3.7_offline.sh

wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/4a5320d1-0b48-458d-9668-fd0e4501208c/intel-oneapi-base-toolkit-2024.2.1.101_offline.sh

sudo sh ./intel-oneapi-base-toolkit-2025.1.3.7_offline.sh -a --silent --cli --eula accept

sudo apt install libgomp1

# Installing Intel Extension for PyTorch 
# Dont set the environment variable for oneAPI, the packages installed using uv is enough for pytorch to work with Intel oneAPI
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

uv pip install intel-extension-for-pytorch oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --index-strategy unsafe-best-match

# To get requirements.txt
uv pip freeze > requirements.txt

```
