import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image

def generate_grid(root_dir="archive (1)/Train", save_path="../models/ucf_crime/dataset_samples.png"):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    fig, axes = plt.subplots(2, 7, figsize=(20, 6))
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        files = glob.glob(os.path.join(cls_dir, "*.png"))
        if not files:
            continue
            
        img_path = random.choice(files)
        img = Image.open(img_path)
        
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"{cls}\n{img.size[0]}x{img.size[1]}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved dataset visualization to {save_path}")

if __name__ == "__main__":
    generate_grid()
