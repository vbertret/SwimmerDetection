from src.color_segmentation import ColorBB
from src.metrics.model_performance import IoU_video
from src.random_forest import RandomForestBB


Color = ColorBB("hsv")

IoU_video("../data/images/Valset", "../data/annotations", Color, debug=True, validation=True)
