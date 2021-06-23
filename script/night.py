import numpy as np
import pandas as pd
from src.color_segmentation import ColorBB
from src.metrics.validation import gridBuilding, gridSearchCV
from src.metrics.model_performance import IoU_video

grid = []

for i in range(-30, 31, 10):
    grid.append({'adjust_pt1': i, 'adjust_pt2': 30})

Color = ColorBB("hsv")

df = gridSearchCV(Color, grid, "../data/images/Valset", "../data/annotations", verbose=True, autosave="newadjust2.csv")

df.to_csv("../data/dataframe/newadjust2.csv")