# KLA Hackathon PS01: AI-Based Restoration of Degraded Images

This repository contains our team's submission for the KLA Hackathon (Problem Statement 01). 
We implemented a **Scaled NAFNet (Nonlinear Activation Free Network)** with VGG-19 Perceptual Loss, CutMix augmentations, and EMA weight stabilization to aggressively hallucinate high-frequency textures (water ripples, brick mortar) that traditional U-Nets destroy.

## Repository Contents
1. `evaluation.py`: Standalone script to run inference on degraded images.
2. `best_nafnet_ema.pt`: The highly optimized model weights.
3. `nafnet_training.ipynb`: Our complete training pipeline (Kaggle notebook).
4. `requirements.txt`: Environment dependencies.

## Setup Instructions

A reviewer can easily clone this repository and run inference on an H100 (or any PyTorch-enabled GPU).

**1. Clone the repository:**
```bash
git clone <YOUR_GITHUB_LINK_HERE>
cd <YOUR_REPO_NAME>
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Evaluation Script

The benchmarking team can use `evaluation.py` to restore degraded `.npy` images exactly as specified in the hackathon rules. 

**Usage:**
```bash
python evaluation.py --input_dir <PATH_TO_TEST_IMAGES> --output_dir <PATH_TO_SAVE_OUTPUTS>
```

**Example:**
```bash
python evaluation.py --input_dir ./data/test/NoisyLR --output_dir ./data/test/Restored
```

### Script Arguments:
* `--input_dir`: Path to the directory containing degraded `.npy` files.
* `--output_dir`: Path to the directory where the restored `256x256` `.npy` images will be saved.
* `--model_path`: (Optional) Path to the weights file. Defaults to `best_nafnet_ema.pt` in the current directory.

## Training
If you wish to review or reproduce our training process, open `nafnet_training.ipynb`. It is configured to run on Kaggle/Colab and contains our Grandmaster training pipeline including:
* Custom D4 & CutMix Data Augmentations
* Composite Loss (L1 + Edge Loss + VGG Perceptual Loss)
* Automatic Mixed Precision (AMP)
* Cosine Annealing Learning Rate
* Exponential Moving Average (EMA) weight stabilization
