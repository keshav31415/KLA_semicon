import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ==========================================
# 1. MODEL ARCHITECTURE (Scaled NAFNet)
# ==========================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.conv2 = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.sg = SimpleGate()
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1)
        )
        self.conv3 = nn.Conv2d(c, c, 1)
        self.norm1 = nn.GroupNorm(1, c)
        
        self.conv4 = nn.Conv2d(c, c * 2, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.norm2 = nn.GroupNorm(1, c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma

class ScaledNAFNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 48
        self.intro = nn.Conv2d(1, c, 3, padding=1)
        
        self.enc1 = nn.Sequential(*[NAFBlock(c) for _ in range(2)])
        self.down = nn.Conv2d(c, c * 2, 2, stride=2)
        self.enc2 = nn.Sequential(*[NAFBlock(c * 2) for _ in range(2)])
        
        self.mid = nn.Sequential(*[NAFBlock(c * 2) for _ in range(2)])
        
        self.up = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = nn.Sequential(*[NAFBlock(c) for _ in range(2)])
        
        self.ending = nn.Conv2d(c, 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x1 = self.intro(x)
        x1 = self.enc1(x1)
        x2 = self.down(x1)
        x2 = self.enc2(x2)
        x2 = self.mid(x2)
        x_up = self.up(x2)
        x_up = x_up + x1
        x_out = self.dec1(x_up)
        out = self.ending(x_out)
        return self.pixel_shuffle(out)

# ==========================================
# 2. EVALUATION PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="KLA Hackathon Evaluation Script (NAFNet)")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to test images directory (containing .npy files)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save restored outputs')
    parser.add_argument('--model_path', type=str, default='best_nafnet_ema.pt', help='Path to trained model weights (.pt file)')
    args = parser.parse_args()

    # 1. Setup Environment & Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load Model
    print(f"[INFO] Loading Scaled NAFNet model from {args.model_path}...")
    model = ScaledNAFNet().to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found at {args.model_path}. Please ensure the file exists.")
        
    # Load weights (map location ensures it works even if tested on CPU machine first)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("[INFO] Model loaded successfully.")
    
    # 3. Find Input Files
    test_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    if len(test_files) == 0:
        print(f"[WARNING] No .npy files found in {args.input_dir}")
        return
        
    print(f"[INFO] Found {len(test_files)} degraded images. Starting inference...")
    
    # 4. Inference Loop
    with torch.no_grad():
        for file_path in tqdm(test_files, desc="Processing Images"):
            filename = os.path.basename(file_path)
            
            # Load and format degraded image
            deg_img = np.load(file_path).astype(np.float32)
            # Add batch and channel dimensions: [1, 1, 128, 128]
            deg_tensor = torch.tensor(deg_img).unsqueeze(0).unsqueeze(0).to(device)
            
            # Run inference (use AMP for maximum speed on H100)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                restored_tensor = model(deg_tensor)
                
            # Post-process back to numpy [256, 256]
            restored_img = restored_tensor.squeeze().cpu().numpy()
            restored_img = np.clip(restored_img, 0.0, 1.0)
            
            # Save Output
            out_path = os.path.join(args.output_dir, filename)
            np.save(out_path, restored_img)
            
    print(f"\\n[SUCCESS] Evaluation complete. All restored images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
