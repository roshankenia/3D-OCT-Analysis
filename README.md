# README for Running OCT Model Training and Testing

## Overview
This repository contains code for training and testing AI-CNet3D: An Anatomically-Informed Cross-Attention Network for Enhanced Glaucoma Detection and Interpretability in 3D OCT Volumes. The model incorporates various attention mechanisms and visualization techniques, and it is designed to work with Zeiss and Topcon datasets.

## Requirements
Before running the training and testing pipeline, ensure that you have the necessary dependencies installed.

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- Monai
- tqdm
- torcheval
- scikit-learn

To install the required dependencies, use the following command:
```bash
pip install torch torchvision opencv-python numpy matplotlib pyyaml monai tqdm torcheval scikit-learn
```

## Running the Code
### Step 1: Activate the Environment
Ensure you activate the correct Python environment before executing the script. If using Conda:
```bash
source /home/roshan/miniconda3/bin/activate DL
```

### Step 2: Define the Configuration File
The training pipeline relies on a YAML configuration file that specifies model parameters, dataset locations, and training settings. Ensure you have an appropriate configuration file (e.g., `./configs/no_aug_no_att.yaml`).

### Step 3: Train and Test the Model
To execute the training and testing process, run:
```bash
bash train_test.sh
```

Alternatively, you can run the Python script directly:
```bash
python train_test.py --config ./configs/no_aug_no_att.yaml
```

## Configuration File Format
The YAML configuration file should define key parameters such as:
```yaml
dataset:
  seeds: [0, 1, 2, 3]
  provider: "Topcon"
  glaucoma_dir: "./data/glaucoma"
  non_glaucoma_dir: "./data/non_glaucoma"
  data_weight: "same"
  data_size: [128, 128, 64]

model:
  augmentation: false
  attention: false
  att_type: "N/A"
  aug_type: "N/A"

experiment:
  batch_size: 8
  num_epochs: 50
  learning_rate: 0.001
  loss: "BCELoss"
  name: "glaucoma_detection"
  patience: 5
  device: "cuda"
```

Modify the configuration file as needed for your experiment.

## Results and Logs
The script will save logs, trained models, and validation metrics in a structured directory. The results will be stored in:
```
./<dataset_provider>_Results/
    ├── <experiment_name>/
    │   ├── <experiment_info>_<data_weight>_<aug>_<att>_<data_size>/
    │   │   ├── trial_<seed>/
    │   │   │   ├── models/ (Trained models saved here)
    │   │   │   ├── data/ (Visualization images saved here)
    │   │   │   ├── results.log (Log output per trial)
    │   ├── results.log (Final aggregated results)
```

