import os
import glob
from collections import defaultdict
from PIL import Image

def get_source_video(filename):
    # Extracts "RoadAccidents003_x264" from "RoadAccidents003_x264_10.png"
    # Or "Abuse028_x264" from "Abuse028_x264_450.png"
    # Actually, split by "_x264" or by last "_"
    parts = filename.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0]
    return filename.split('.')[0]

def analyze_dataset(root_dir="archive (1)"):
    train_dir = os.path.join(root_dir, "Train")
    test_dir = os.path.join(root_dir, "Test")
    
    train_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    test_classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    print("==================================================")
    print("1. CLASS MAPPING")
    print("==================================================")
    print(f"Train classes ({len(train_classes)}): {train_classes}")
    print(f"Test classes ({len(test_classes)}): {test_classes}")
    if train_classes != test_classes:
        print("CRITICAL ERROR: Train and Test classes do not match!")
    else:
        print("Class mapping is consistent.")
        
    print("\n==================================================")
    print("2. CLASS DISTRIBUTION & SOURCE VIDEOS")
    print("==================================================")
    
    train_sources = set()
    test_sources = set()
    
    train_dist = {}
    test_dist = {}
    
    def process_split(split_dir, classes, global_sources_set):
        dist = {}
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            files = glob.glob(os.path.join(cls_dir, "*.png"))
            
            # Analyze source videos
            cls_sources = set()
            for f in files:
                basename = os.path.basename(f)
                src = get_source_video(basename)
                cls_sources.add(src)
                global_sources_set.add(src)
                
            dist[cls] = {
                "num_frames": len(files),
                "num_sources": len(cls_sources),
                "example_file": files[0] if files else None
            }
        return dist
        
    train_dist = process_split(train_dir, train_classes, train_sources)
    test_dist = process_split(test_dir, test_classes, test_sources)
    
    print(f"{'Class':<20} | {'Train Frames':<15} | {'Train Videos':<15} | {'Test Frames':<15} | {'Test Videos':<15}")
    print("-" * 85)
    for cls in train_classes:
        tr = train_dist.get(cls, {"num_frames": 0, "num_sources": 0})
        te = test_dist.get(cls, {"num_frames": 0, "num_sources": 0})
        print(f"{cls:<20} | {tr['num_frames']:<15} | {tr['num_sources']:<15} | {te['num_frames']:<15} | {te['num_sources']:<15}")
        
    print("\n==================================================")
    print("3. SOURCE VIDEO OVERLAP (LEAKAGE)")
    print("==================================================")
    overlap = train_sources.intersection(test_sources)
    print(f"Unique Train Videos: {len(train_sources)}")
    print(f"Unique Test Videos: {len(test_sources)}")
    print(f"Videos in BOTH Train and Test (Leakage): {len(overlap)}")
    if len(overlap) > 0:
        print("WARNING: Data leakage detected! The following videos are in both splits:")
        for v in list(overlap)[:20]:
            print(f" - {v}")
        if len(overlap) > 20:
            print(f" ... and {len(overlap) - 20} more.")
            
    print("\n==================================================")
    print("4. IMAGE FORMAT & DIMENSIONS")
    print("==================================================")
    example_img = train_dist[train_classes[0]]["example_file"]
    if example_img:
        try:
            with Image.open(example_img) as img:
                print(f"Format: {img.format}")
                print(f"Mode (RGB/RGBA): {img.mode}")
                print(f"Dimensions: {img.size}")
        except Exception as e:
            print(f"Error reading image: {e}")

if __name__ == "__main__":
    analyze_dataset()
