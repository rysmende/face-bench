import math
import torch.nn as nn
import torch
import numpy as np

from ml_models.mtcnn import MTCNN
from ml_models.lightcnn import LightCNN
import torchvision as tv
from PIL import Image
from ml_models.facerec_utils import postprocess_face, check_image_and_box

class FaceRec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mtcnn = MTCNN()
        self.lightcnn = LightCNN()
        self.image_processing = tv.transforms.Compose([
            tv.transforms.Resize((128, 128)),
            tv.transforms.ToTensor()
        ])

    def forward(self, full_image: Image.Image, device):
        x = torch.from_numpy(np.array(full_image)).to(device)
        y, _ = self.mtcnn(x)
        boxes, probs = postprocess_face(y)
        res = check_image_and_box(full_image, boxes, probs)
        if res != 0:
            return None, res - 1
        box = boxes[0]
        face = full_image.crop(box)
        face = self.image_processing(face)
        # Face Embeddings
        x = face.unsqueeze(0).to(device)
        y = self.lightcnn(x)
        y = y.squeeze()
        return y, 0

