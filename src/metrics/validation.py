import numpy as np
import cv2.cv2 as cv
import pandas as pd
from src.metrics.model_performance import IoU_video


def gridBuilding(parameters):
    parameters_values = []

    for i, grid in enumerate(parameters):
        if i == 0:
            for values in parameters[grid]:
                parameters_values.append({grid: values})
        else:
            cp = parameters_values.copy()
            parameters_values.clear()
            for values in parameters[grid]:
                cp2 = cp.copy()
                for j in range(len(cp2)):
                    cp2[j] = cp2[j].copy()
                    cp2[j][grid] = values
                parameters_values += cp2
    return parameters_values

def gridSearchCV(model, grid, dir_name, dir_annot, verbose=False, autosave=""):

    df = pd.DataFrame()

    for params in grid:
        model.set_params(**params)

        if verbose:
            print(params)

        _, _, stat_values = IoU_video(dir_name, dir_annot, model, validation=True)

        if verbose:
            print("mean IoU : ", stat_values[-1][0], " Number of images with no box : ", stat_values[-1][1],
                  " IoU mean with no box : ", stat_values[-1][2], " mean score : ", stat_values[-1][3])

        index = ""
        for key, values in params.items():
            index += str(key) + " : " + str(values) + ", "

        index = [index[0:-2]]

        V1 = pd.DataFrame(np.array(stat_values[0]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)
        V2 = pd.DataFrame(np.array(stat_values[1]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)
        T3 = pd.DataFrame(np.array(stat_values[2]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)
        Total = pd.DataFrame(np.array(stat_values[3]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)

        row = pd.concat([Total, V1, V2, T3], axis=1, keys=('Total', 'V1', 'V2', 'T3'))

        df = df.append(row)

        if autosave != "":
            df.to_csv(f"data/dataframe/{autosave}")

    return df


