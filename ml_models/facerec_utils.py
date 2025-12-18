import numpy as np
from PIL import Image
def postprocess_face(boxes: np.ndarray) -> tuple:
    '''
    Sort boxes by its area and extract probabilities.

        Parameters:
            boxes (np.ndarray): Coordinates of bounding boxes
        Returns:
            boxes (np.ndarray): Coordinates of bounding boxes
            probs (np.ndarray): Probabilities of bounding boxes
    '''
    b = boxes[0]
    b = b[np.argsort((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[::-1]]
    boxes = b[:, :4].copy()
    probs = b[:,  4].copy()
    return boxes, probs

def check_image_and_box(img: Image.Image, boxes: np.ndarray, score: np.ndarray) -> int:
    '''
    Check image and face coordinates for errors.

        Parameters:
            img (Image.Image): Whole input image
            boxes (np.array): Coordinates of bounding boxes
            score (np.array): Probabilities of each bounding box
        Returns:
            result (int): Code result of checking 
    '''
    w, h = img.size

    # if (h < w):
    #     return 5

    if boxes is None:
        return 2

    # Filter by scores.
    idx = score > 0.85
    boxes  = boxes[idx]
    
    # Filter by area.
    # areas = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes])
    # img_area = h * w
    # area_idx = (areas / img_area) > 0.1
    # boxes  = boxes[area_idx]
    
    if len(boxes) < 1:
        return 2 # Zero faces or too small.
    
    if len(boxes) > 1:
        return 3 # More than one face on image.
    
    # Get the first element from the list.
    box   = boxes[0]
    
    x0, y0, x1, y1 = box # Get coordinates.
    
    # Check coordinates to align with image's dimensions.
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return 4 # Face is not centered.
    
    return 0