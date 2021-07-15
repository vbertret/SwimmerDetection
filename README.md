# SwimmerDetection

This project was done during an internship within  the  VAADER  (Video  Analysis  and  Architecture  Design  for Embedded 
Resources) team in IETR Lab.
The internship is part of a project financed by DIGISPORT whose goal is the realization of
a system for monitoring a swimming activity in a swimming pool,
based on image and video processing algorithms and artificial intelligence.

# :swimming_man: Description of the project

The main purpose of the project is to study methods of tracking a swimmer in a pool 
by embedded active vision in order to understand the swimming movement.

During  my  internship,  I  was  focused  on  the  detection  of  the  swimmer  body  in  the
entire image  with  the  use  of  different  image  processing  algorithms  in  order  to
extract  the  area  of  the image  containing  the  region  of  interest.   This  area  is  then
transferred  to  the  central  processing system which performs a pose detection in order to analyse
the swimmer’s movement.

I was in charge of the study and comparison in simulation of active vision algorithms allowing the localization of the swimmer in the image.

There 3 different algorithms :
* Color Segmentation
* Gaussian Mixture
* Deep Learning

The projects also contains a database of more than 2,600 images of swimmer with bounding box. 

If you want to test by yourself, you have to fork the repository. After, you just have to execute the 2 commands to install all the packages needeed for the project and especially the package **src** :  

```
pip install -r requirements.txt
pip install -e .
```

⚠️If you want Pytorch with cuda for GPU, you have to install it by your own.

# :book: How to use the package src

If you want to test the model directly on the database, here are some guidelines :

* Firstly, import the model you want to use and the method **IoU_video** to see the results :
```
from src.color_segmentation import ColorBB
from src.gaussian_mixture import GaussianMixtureBB
from src.deep_learning import Swimnet
from src.metrics.model_performance import IoU_video
```

* Then, load the model :
```
#### Gaussian Mixture ####

filename = "../models/GMM_model_test_vid_23_full"
model = GaussianMixtureBB(filename, threshold=0.5, graph_cut=False, use_time=False)

#### Deep Learning ####

filename = "../models/mobilenet-V3-small"
model = Swimnet("mobilenet-v3-small")
model.load_state_dict(torch.load(filename))

#### Color Segmentation ####

model = ColorBB("hsv", use_time=False)
```

* Finally, call the method **IoU_video** and specify the location of the database

```
IoU_values, stat_values = IoU_video("../data/images/Testset", "../data/annotations", model, debug=True)
```

If you want more details about the parameters of each model, there are well
defined in the documentation of each class.

# Presentation of the different method

## Color Segmentation
![SwimXYZ Presentation](https://user-images.githubusercontent.com/6428515/125781660-929177f3-3bd2-413a-a26e-dde4c466bf8d.jpg)

## Gaussian Mixture
![SwimXYZ Presentation (1)](https://user-images.githubusercontent.com/6428515/125781810-79f947bd-8262-48f1-ab7c-65e626766422.jpg)

## Swimnet
![SwimXYZ Presentation (2)](https://user-images.githubusercontent.com/6428515/125781890-1020594b-a976-4a20-a840-d661a0eea74d.jpg)

# Organisation
* `\data`-- all the data of the project
  * `\images` all the images of the database separated in 3 directories Testset, Trainset and Valset
  * `\annotations` all the annotations of the database organized as in images
  * `\dataframes` all the results of validation tests to find the best parameters
  * `\figures` figures for the report
* `\docs` some documents made for the project
* `\models` all the saved models
* `\notebooks` different notebooks with example of code to illustrate the projects and to have some example of the syntax in order to use the package
* `\reports` the report made for the project
* `\src` the package developed for the project. Each method or class is documented
  * `\annotations` 
    * `read_annotations.py` methods to read annotations of the database
  * `\metrics`
    * `model_performance.py` methods to compute metrics and to compare models
    * `validation.py` methods to make validation
  * `\preporcessing`
    * `bb_tools.py` some tools for bouding box
    * `dataset.py` all the class and methods in order to create the dataset
    * `water_surface_detection.py` methods to detect the surface
  * `color_segmentation.py` class for the Color segmentation
  * `deep_learning.py` class for the Deep Learning models
  * `gaussian_mixture.py` class for the Gaussian Mixture model
  * `kalman.py` implementation of the Kalman Filter
* `\scripts` scripts which used the packages
* `requirements.txt` all the package you need to install so that you can use the project on your own
