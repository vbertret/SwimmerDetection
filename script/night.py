import numpy as np
import pandas as pd
from Code.ColorSegmentation import ColorBB
from Code.Metrics.Validation import gridBuilding, gridSearchCV
from Code.Metrics.ModelPerformance import IoU_video

grid = []

for k in range(70, 91, 10):
    for i in range(140, 161, 10):
        for j in range(70, 91, 10):
            grid.append({'upper_yuv': [k, i, 255], 'lower_yuv': [0, 0, j]})

Color = ColorBB("yuv", adjust_pt1=0, adjust_pt2=30)

df = gridSearchCV(Color, grid, "../Images/Valset", "../Annotations", verbose=True, autosave="yuvbound2.csv")

df.to_csv("../Data/yuvbound2.csv")