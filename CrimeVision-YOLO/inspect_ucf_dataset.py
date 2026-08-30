import os
import glob
from collections import defaultdict

def inspect_dataset(base_path):
    print("==================================================")
    print("UCF CRIME DATASET INSPECTION")
    print("==================================================")
    
    if not os.path.exists(base_path):
        print(f"Error: {base_path} not found.")
        return
        
    for split in ['Train', 'Test']:
        split_dir = os.path.join(base_path, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found.")
            continue
            
        print(f"\n--- {split} Split ---")
        classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        print(f"Total Classes: {len(classes)}")
        
        total_files = 0
        extensions = defaultdict(int)
        
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
            file_count = len(files)
            total_files += file_count
            
            # Count extensions
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                extensions[ext] += 1
                
            print(f"  {cls}: {file_count} files")
            
        print(f"Total {split} Samples: {total_files}")
        print(f"Extensions Found: {dict(extensions)}")

if __name__ == "__main__":
    inspect_dataset("/Users/saikrishivgars/Desktop/YOLOO/archive (1)")
