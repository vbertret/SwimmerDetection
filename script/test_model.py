from src.color_segmentation import ColorBB
from src.gaussian_mixture import GaussianMixtureBB
from src.deep_learning import Swimnet
from src.combining_methods import CombiningBB
from src.metrics.model_performance import IoU_video
import torch

#### Gaussian Mixture ####

filename = "../models/GMM_model_test_vid_23_full"
model = GaussianMixtureBB(filename, threshold=0.01, use_time=False, graph_cut=False)

#### Deep Learning ####

PATH = "../models/dataaug6_end"
model = Swimnet("mobilenet-v3-small")
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

#### Color Segmentation ####

model = ColorBB("hsv", use_time=True)
model = ColorBB("yuv", use_time=True)

#### Combining model ####

model = CombiningBB(use_time=True)

#### Test the model ####

IoU_video("../data/images/Testset", "../data/annotations", model, debug=True)

