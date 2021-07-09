from src.color_segmentation import ColorBB
from src.gaussian_mixture import GaussianMixtureBB
from src.deep_learning import Swimnet
from src.metrics.model_performance import IoU_video
import torch

#### Gaussian Mixture ####

#filename = "../models/GMM_model_test_vid_23_full"
#model = GaussianMixtureBB(filename, threshold=0.5, graph_cut=False, use_time=False)

#### Deep Learning ####

# PATH = "../models/mobilenet-V3-small"
# model = Swimnet("mobilenet-v3-small")
# model.load_state_dict(torch.load(PATH))

#### Color Segmentation ####

model = ColorBB("hsv", use_time=False)

IoU_values, stat_values = IoU_video("../data/images/Testset", "../data/annotations", model, debug=True)

print(IoU_values)
