import cv2.cv2 as cv
import numpy as np
import time
from Code.Preprocessing.BBTools import union, intersection
from Code.Preprocessing.WaterSurfaceDetection import surface_detection
from Code.Annotations.read_annotation import read_annotation


def IoU(box1, box2):
    """
    Intersection over Union

    Compute the Intersection over Union of the two boxes

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the first rectangle box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the second rectangle box2

    Returns
    -------
    IoU_val : float
        Intersection over Union
    """

    # Computation of the areas of box1 and box2
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]

    # Construction of the intersection between box1 and box2
    intersection_box = intersection(box1, box2)

    # Computation of the area of the intersection
    area_intersection = intersection_box[2] * intersection_box[3]

    # Computation of the Intersection over Union
    IoU_val = area_intersection / (area_box1 + area_box2 - area_intersection)

    return IoU_val


def IoU_video(dir_name, dir_annot, model, threshold=0.5, choice=None, debug=False):
    """
    Computation of the IoU for several videos

    The method computes the IoU for each frame of the video in the directory dirname. The user has to specify
    the global directory of the annotation and the images and annotations directory need to be organized exactly
    in the same way. Finally, the directory dirname must contains a file info.txt with informations about the videos.

    Parameters
    ----------
    dir_name : str
        the directory name of the directory containing the frames
    dir_annot : str
        the name of the global directory of the annotations
    model : class with a predict function
        the model
    threshold : float
        the number use for the prediction of some models
    choice : list
        the list of the videos of interest ( default is None )
    debug : boolean
        if true, the method displays each step ( default is False )

    Returns
    -------
    IoU_values : list of list
        it contains all the IoU values for all the frames and all the videos
    stat_values : list of list
        it contains some statistics about each videos : mean IoU, the number of frames with the IoU equal to 0,
        and the mean IoU without the frames with IoU equal to 0
    """
    # Reading the info.txt file
    f = open(f"{dir_name}/info.txt")
    data = f.readlines()
    f.close()

    # Preprocessing to have a structured format
    data_videos = [line.split(";") for line in data]
    data_videos = [[line[0], int(line[1]), int(line[2]), line[3].replace("\n", "")] for line in data_videos]

    # Choice of the videos on which the IoU will be computed
    if choice==None:
        queue = range(len(data_videos))
    else:
        queue = [i-1 for i in choice]

    # Initialization of some tables and values
    IoU_values = []
    stat_values = []
    t_ite = time.time()
    quit = False
    stat_total = [0, 0, 0]
    nb_total = 0

    # Computation of the IoU for all the videos in Queue
    for i in queue:

        # Retrieving of the video title and the range the values of the frames
        video_title = data_videos[i][0]
        start = data_videos[i][1]
        end = data_videos[i][2]
        nb = end-start

        if debug:
            print(f"Video Treated : {video_title}...")

        # If precised, the surface is detected
        if model.detect_surface:
            a, b = surface_detection(f"{dir_name}/background/{data_videos[i][3]}", 117, adjust_pt1=model.adjust_pt1, adjust_pt2=model.adjust_pt2)
        else:
            a, b = 0, 0

        # Initialization of the bounding box and the array of IoU values
        BB = []
        IoU_video = []
        no_box = 0

        # Computation of the IoU for all the frames of the video
        for nb_filename in range(start, end):

            # Declaration of the filename of the frames which is being treated
            file_name = dir_name + "/" + str(video_title) + str(nb_filename).zfill(5) + ".jpg"

            # Declaration of the filename of the annotations which is being treated
            annot_name = dir_annot + "/" + dir_name.split("/")[-1] + "/" + str(video_title) + str(nb_filename).zfill(5) + ".json"

            # Prediction of the bounding box. If precised, the method gives to the
            # predict function, the previous predicted bounding box
            if model.use_time:
                BB = model.predict(file_name, a=a, b=b, precBB=BB)
            else:
                BB = model.predict(file_name, a=a, b=b, precBB=[])

            # Computation of the number of frames per second
            fps = 1 / (time.time() - t_ite)

            # Read the ground truth annotation
            BB_ground_truth = read_annotation(annot_name)

            if (debug):
                img = cv.imread(file_name, cv.IMREAD_COLOR)
                x_a, y_a, w_a, h_a = BB_ground_truth
                cv.rectangle(img, (x_a, y_a), (x_a + w_a, y_a + h_a), (255, 0, 0), 2)

            # If there is no bounded box found, the IoU is equal to O
            if BB == []:
                IoU_video.append(0)
                no_box += 1
            # Otherwise, the method compute the IoU
            else:
                IoU_video.append(IoU(BB, BB_ground_truth))

                if IoU_video[-1] == 0:
                    no_box += 1

                if debug:
                    x, y, w, h = BB
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(img, "Predicted BB", (x, y + h + 20), 0, 0.5, (0, 255, 0), 2)
                    cv.putText(img, "IoU : " + str(round(IoU_video[-1], 2)), (20, 40), 0, 1.5, (0, 0, 255), 2)
                    cv.putText(img, "True BB", (x_a, y_a - 10), 0, 0.5, (255, 0, 0), 2)

            if debug:
                cv.putText(img, "FPS : " + str(round(fps, 2)), (350, 40), 0, 1.5, (0, 255, 255), 2)
                cv.imshow(f"Bounding box", img)
                if cv.waitKey(2) & 0xFF == ord('q'):
                    quit = True
                    break

            # Time computation for the computation of the FPS
            t_ite = time.time()

        # Put the values into a list and compute some results to display
        IoU_values.append(IoU_video)
        stat_values.append([np.sum(IoU_video) / nb, no_box, np.sum(IoU_video) / (nb - no_box)])
        stat_total[0] = stat_total[0] + np.sum(IoU_video)
        stat_total[1] = stat_total[1] + no_box
        stat_total[2] = stat_total[2] + np.sum(IoU_video)
        nb_total += nb

        if debug:
            print("mean IoU : ", stat_values[-1][0], " Number of images with no box : ", stat_values[-1][1],
                " IoU mean with no box : ", stat_values[-1][2])

        if quit:
            break

    stat_total[0] = stat_total[0]/nb_total
    stat_total[2] = stat_total[2]/(nb_total - stat_total[1])
    stat_values.append(stat_total)
    cv.destroyAllWindows()

    return IoU_values, stat_values

