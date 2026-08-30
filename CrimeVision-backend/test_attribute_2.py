import sys
sys.path.append('../CrimeVision-YOLO')
from attribute import AttributeExtractor
import numpy as np
extractor = AttributeExtractor()
print("Extractor initialized.")
crop = np.ones((100, 100, 3), dtype=np.uint8) * 255
try:
    res = extractor.describe_crop(crop)
    print("Describe crop returned:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
