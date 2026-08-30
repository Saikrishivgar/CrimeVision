import sys
sys.path.append('../CrimeVision-YOLO')
try:
    from attribute import AttributeExtractor
    import cv2
    import numpy as np
    from PIL import Image
    extractor = AttributeExtractor()
    print("Extractor initialized.")
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = cv2.imread("../CrimeVision-backend/uploads/annotated_8a15169e28ac4e67b7258cea25500c70.mp4") # just a dummy if needed, but let's use a real image
    if crop is None:
        crop = np.ones((100, 100, 3), dtype=np.uint8) * 255
    res = extractor.describe_crop(crop)
    print("Describe crop returned:", res)
except Exception as e:
    import traceback
    traceback.print_exc()



    