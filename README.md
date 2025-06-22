# 🌱 Plant Disease Identifier with PyTorch
Help identify plant diseases - Focus on sugarcane and rice in indian context using Intel oneAPI and PyTorch. It also showcases how to set up a PyTorch project with Intel oneAPI for better performance on Intel hardware.

## Key Features
- **Plant Disease Classification**: Identify diseases in sugarcane and rice plants.
- **Intel oneAPI Integration**: Leverage Intel's oneAPI for optimized performance on Intel hardware.
- **PyTorch Framework**: Utilizes PyTorch for deep learning model development.
- **Easy Setup**: Simple setup instructions for getting started with the project.
- **Performance Comparison**: Show case the performace gains using XPU vs CPU

## Project Structure
```.
├── .venv/                          # Python virtual environment (not tracked by git)
├── config/                         # Configuration files for the project
│   └── config.py                   # Main configuration file for the project
├── data/                           # All datasets (raw and processed)
│   ├── raw/                        # Raw images (unprocessed)
│   │   ├── sugarcane_images/       # Raw sugarcane images
│   │   └── rice_images/            # Raw rice images
│   └── processed/                  # Processed/split datasets for training/validation/testing
│       ├── sugarcane/              # Processed sugarcane data
│       └── rice/                   # Processed rice data
├── models/                         # Saved PyTorch model weights and checkpoints
├── scripts/                        # Training and inference scripts (entry points)
│   ├── rice_train_model.py         # Training script for rice disease model
│   └── sugarcane_train_model.py    # Training script for sugarcane disease model
├── src/                            # Source code for the project
│   ├── data/                       # Data loading, preparation, and augmentation modules
│   │   ├── data_analysis.py        # Data analysis and visualization scripts
│   │   ├── data_augmentation.py    # Data augmentation utilities
│   │   └── data_preparation.py     # Data preparation scripts
│   ├── inference/                  
│   │   └── predictor.py            # Config driven inference script for making predictions
│   ├── models/                     # Model architecture definitions
│   │   └── model_architecture.py   # Model architecture definitions with required layers
│   └─ training/                    # Training utilities and loss functions
│       └── trainer.py              # Loading the model into right device and training the model
├── requirements.txt                # Python package dependencies
├── pyproject.toml                  # Project configuration for uv
├── README.md                       # Project documentation (this file)
└── mlruns/                         # MLflow experiment tracking and artifacts

```
## How to run the project
### Configurations
Update the DEVICE in `config/config.py` to either `cpu` or `xpu:0` to use integrated Arc GPU or not.
> **_NOTE_**: Instead of having two different code, use the DEVICE configuration to switch between CPU and XPU. Since we anyway need to run the code using `cpu` or `gpu(xpu:0)`

### Setup the environment

```bash
# Install uv latest version
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize a new project (if not already done, creates pyproject.toml)
uv init

# Create a virtual environment (e.g., named.venv) using a specific Python version
uv venv

# Activate the virtual environment
source .venv/bin/activate

```

### Intel libraries
Installation of Intel oneAPI and PyTorch with Intel Extension

```bash

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

## Training the model
- Enable `cpu` or `xpu:0` in `config/config.py` to switch between CPU and XPU.

```bash
# Activate the virtual environment if not already done
source .venv/bin/activate

# Train the rice disease model
python scripts/rice_train_model.py

# Train the sugarcane disease model
python scripts/sugarcane_train_model.py
```

- Trained mode is availabe in the folder `models/` with the name `rice_model.pth` and `sugarcane_model.pth`.
- All performance information is logged in the MLFlow

### Viewing the MLFlow
```bash
# Activate the virtual environment if not already done
source .venv/bin/activate
# Start the MLFlow server
mlflow ui --port 5000
```
- Open your web browser and go to `http://localhost:5000` to view the MLFlow UI.
- You can view the experiments, metrics, and artifacts logged during training.
- Look for the run metrics to compare the performance of CPU vs XPU training.
  - `epoch_time_sec` - Time taken for each epoch
  - `total_time_sec` - Total time taken for training
  - `accuracy` - Accuracy of the model on the validation set
