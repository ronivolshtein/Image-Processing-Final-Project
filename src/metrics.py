import numpy as np

def compute_iou_bbox(boxA, boxB, epsilon=1e-5):
    """
    Computes the IoU (Intersection over Union) metric between two bounding boxes.
    Input format expected: [xmin, ymin, xmax, ymax]
    """
    x1 = max(boxA, boxB)
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    width = (x2 - x1)
    height = (y2 - y1)

    # If the boxes do not overlap at all
    if (width < 0) or (height < 0):
        return 0.0

    area_overlap = width * height
    area_a = (boxA[2] - boxA) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB) * (boxB[3] - boxB[1])
    
    # Calculate the combined area (Union)
    area_combined = area_a + area_b - area_overlap

    # Calculate Intersection divided by Union
    iou = area_overlap / (area_combined + epsilon)
    
    return float(iou)