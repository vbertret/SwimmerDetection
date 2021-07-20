import numpy as np
import pandas as pd
from src.color_segmentation import ColorBB
from src.gaussian_mixture import GaussianMixtureBB
from src.metrics.validation import gridBuilding, gridSearchCV
from src.metrics.model_performance import IoU_video

# First way to create the grid
grid = []
for i in range(-50, 51, 10):
    for j in range(-50, 51, 10):
            grid.append({'adjust_pt1': i, "adjust_pt2" : j})

# Second way to create the grid
parameters = {"adjust_pt1" : np.arange(-50, 51, 10), "adjust_pt2" : np.arange(-50, 51, 10)}
grid = gridBuilding(parameters)

# Definition of the model
model = ColorBB(detect_surface=True, use_time=False)

# Exhaustive validation search
savefile = None
df = gridSearchCV(model, grid, "data/images/Valset", "data/annotations", verbose=True, autosave=savefile)
df.to_csv(f"data/dataframe/{savefile}")