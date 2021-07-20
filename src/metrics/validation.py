import numpy as np
import cv2.cv2 as cv
import pandas as pd
from src.metrics.model_performance import IoU_video


def gridBuilding(parameters):
    """
    Create a grid of parameters in order to make a validation search after.

    The methods takes a dictionnary of possibility for each paramaters as inputs and 
    creates a grid in order to test each combinaison of parameters.

    Parameter
    ----------
    params : dict
        All possibilities of parameters

    Return
    ------
    parameters_values
        A grid with one row for each combinaison of parameters
    """

    # Initialize the grid 
    parameters_values = []

    # Build the grid
    for i, grid in enumerate(parameters):
        # If it is the first parameter we just add one row for each possibility
        if i == 0:
            for values in parameters[grid]:
                parameters_values.append({grid: values})
        # Otherwise, for each existing row, create new rows for each possibility
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
    """
    Exhaustive search over specified parameter values for an estimator.

    The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

    Parameters
    ----------
    model : 
        This is assumed to implement the estimator interface. 
        Either estimator needs to provide a predict function.
    grid : dict
        Dict created by the method gridBuilding
    dir_name : str
        the directory name of the directory containing the frames
    dir_annot : str
        the name of the global directory of the annotations
    verbose : bool
        If true, a summary of each test is printed ( default is False )  
    autosave : str
        The filename where the experience can be saved ( default is "" )      
    
    Return
    ------
    df : pandas.Dataframe
        The dataframe with all the results
    """

    # Initialize the dataframe
    df = pd.DataFrame()

    # Iterate over all the experiments of the grid
    for params in grid:
        model.set_params(**params)

        if verbose:
            print(params)

        # Compute the score for the experiment
        _, _, stat_values = IoU_video(dir_name, dir_annot, model, validation=True)

        if verbose:
            print("mean IoU : ", stat_values[-1][0], " Number of images with no box : ", stat_values[-1][1],
                  " IoU mean with no box : ", stat_values[-1][2], " mean score : ", stat_values[-1][3])

        # Concatenate all the parameters of the experiment to be the title of the row
        index = ""
        for key, values in params.items():
            index += str(key) + " : " + str(values) + ", "

        index = [index[0:-2]]

        # Organize the results
        V1 = pd.DataFrame(np.array(stat_values[0]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)
        V2 = pd.DataFrame(np.array(stat_values[1]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)
        T3 = pd.DataFrame(np.array(stat_values[2]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)
        Total = pd.DataFrame(np.array(stat_values[3]).reshape((1, 4)), columns=["Mean IoU", "No box", "Mean IoU adjusted", "Mean score"], index=index)

        # Concatenation of the results
        row = pd.concat([Total, V1, V2, T3], axis=1, keys=('Total', 'V1', 'V2', 'T3'))

        # Concatenation with the results of the precedent experiments
        df = df.append(row)

        # IF specified, save the dataframe after each experiment
        if autosave != "":
            df.to_csv(f"data/dataframe/{autosave}")

    return df


