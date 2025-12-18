from torch import nn
from ml_models.mtcnn_modules import PNet, RNet, ONet
from ml_models.detect_face import detect_face

class MTCNN(nn.Module):
    def __init__(self) -> None:
        super(MTCNN, self).__init__()
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, x):
        return detect_face(x, self.pnet, self.rnet, self.onet)