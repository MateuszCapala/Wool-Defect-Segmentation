import cv2
import numpy as np

def post_process_mask(mask, min_area=100):
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask