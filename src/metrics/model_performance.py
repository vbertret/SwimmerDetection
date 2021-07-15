from os import stat
import cv2.cv2 as cv
import numpy as np
import time
from src.preprocessing.bb_tools import union, intersection
from src.preprocessing.water_surface_detection import surface_detection
from src.annotations.read_annotation import read_annotation
from src.kalman import KalmanFilter


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


def score(box1, box2):
    """
    Implement a score function for the swimmer detection task. The first box is the ground-truth
    and the second box is the predicted.

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the ground-truth rectangle box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the predicted rectangle box2

    Returns
    -------
    score : float ( [-1, 1])
        score value
    """
    # Construction of the intersection between box1 and box2
    intersection_box = intersection(box1, box2)

    # Computation of the IoU
    IoU_val = IoU(box1, box2)

    # Computation of the score
    if intersection_box[0] == box1[0] and intersection_box[1] == box1[1] and intersection_box[2] == box1[2] and intersection_box[3] == box1[3]:
        return IoU_val
    else:
        return IoU_val - 1


def accuracy(box1, box2):
    """
    Accuracy

    Compute the percentage of well classified pixels according to the 2 boxes

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the first rectangle box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the second rectangle box2

    Returns
    -------
    acc : float
        Accuracy
    """
    if len(box1) != 0 and len(box2) != 0:
        # Convert the coordinates into integer if there are floats
        box1 = np.array(box1, dtype=np.int64)
        box2 = np.array(box2, dtype=np.int64)

        # Compute the areas of the box2 and box1
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]

        # Compute the number of well predicted foreground pixels
        inter = intersection(box1, box2)
        area_inter = inter[2]*inter[3]

        # Compute the accuracy
        acc = 480*640 - (area1 - area_inter) - (area2 - area_inter)
    else:
        if len(box1) != 0:
            # Convert the coordinates into integer if there are floats
            box1 = np.array(box1, dtype=np.uint64)

            # Compute the areas of the box1
            area1 = box1[2] * box1[3]

            acc = 480*640 - area1
        else:
            # Convert the coordinates into integer if there are floats
            box2 = np.array(box2, dtype=np.uint64)

            # Compute the areas of the box2
            area2 = box2[2] * box2[3]

            acc = 480 * 640 - area2

    return acc/(640*480)


def precision(box1, box2):
    """
    Precision

    Compute the precision of the prediction. The first box is the ground-truth
    and the second box is the predicted.

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the ground-truth box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the predicted box2

    Returns
    -------
    prec : float
        Precision
    """
    if len(box2) != 0:
        # Convert the coordinates into integer if there are floats
        box1 = np.array(box1, dtype=np.int64)
        box2 = np.array(box2, dtype=np.int64)

        # Compute the areas of the box2 and box1
        area2 = box2[2] * box2[3]

        # Compute the number of well predicted foreground pixels
        inter = intersection(box1, box2)

        if inter[2] == 0 and inter[3] == 0:
            prec = 0
        else:
            TP = inter[2]*inter[3] # True positives

            # Compute the precision
            FP = area2 - TP
            prec = TP/(FP+TP)
    else:
        prec = 0

    return prec


def recall(box1, box2):
    """
    Recall

    Compute the recall of the prediction. The first box is the ground-truth
    and the second box is the predicted.

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the ground-truth box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the predicted box2

    Returns
    -------
    rec : float
        Precision
    """
    if len(box2) != 0:
        # Convert the coordinates into integer if there are floats
        box1 = np.array(box1, dtype=np.int64)
        box2 = np.array(box2, dtype=np.int64)

        # Compute the areas of the box2 and box1
        area1 = box1[2] * box1[3]

        # Compute the number of well predicted foreground pixels
        inter = intersection(box1, box2)

        if inter[2] == 0 and inter[3] == 0:
            rec = 0
        else:
            TP = inter[2]*inter[3] # True positives

            # Compute the precision
            FN = area1 - TP
            rec = TP/(FN+TP)
    else:
        rec = 0

    return rec


def f1_score(box1, box2):
    """
    F1-Score

    Compute the f1-score of the prediction. The first box is the ground-truth
    and the second box is the predicted.

    Parameters
    -----------
    box1: (x1, y1, w1, h1) tuple
        parameters of the ground-truth box1
    box2: (x2, y2, w2, h2) tuple
        parameters of the predicted box2

    Returns
    -------
    f1 : float
        Precision
    """
    if len(box2) != 0:
        # Compute the precision and the recall
        prec = precision(box1, box2)
        rec = recall(box1, box2)

        # Compute the f1-score
        if rec != 0 and prec != 0:
            f1 = 2*(prec*rec)/(prec+rec)
        else:
            f1 = 0

    else:
        f1 = 0

    return f1


def IoU_video(dir_name, dir_annot, model, use_kalman=None, choice=None, debug=False, validation=False, margin=0):
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
    use_kalman : str
        if different of None, a kalman filter is fitted ( default is None )
        The value can be "avg" or "max" if we want to use the exponentially weighted average or the maximum value
        founded for h and w.
    choice : list
        the list of the videos of interest ( default is None )
    debug : boolean
        if true, the method displays each step ( default is False )
    validation : boolean
        if true, the method takes directly tyhe value of a and b already calculated ( default is False )

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
    score_values = []
    stat_values = []
    t_ite = time.time()
    quit = False
    stat_total = [0, 0, 0, 0]
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
            if validation:
                if video_title == "V1":
                    a, b = 0.0328125, 150.0
                elif video_title == "V2":
                    a, b = 0.165625, 184.0
                elif video_title == "T3":
                    a, b = 0.1515625, 198.0
            else:
                a, b = surface_detection(f"{dir_name}/background/{data_videos[i][3]}", 117, adjust_pt1=model.adjust_pt1, adjust_pt2=model.adjust_pt2)
        else:
            a, b = 0, 0

        # Initialization of the bounding box and the array of IoU values
        BB = []
        IoU_video = []
        score_video = []
        no_box = 0

        # Initialization of the value for the KalmanFilter
        if use_kalman is not None:
            found = False
            middle_x = 0
            middle_y = 0
            t=1
            if use_kalman == "avg":
                beta = 0.95
                S_w = 0
                S_h = 0
            elif use_kalman == "max":
                max_h = []
                max_w = []

        # Computation of the IoU for all the frames of the video
        for nb_filename in range(start, end):

            # Declaration of the filename of the frames which is being treated
            file_name = dir_name + "/" + str(video_title) + str(nb_filename).zfill(5) + ".jpg"
            img = cv.imread(file_name)
            img_ini = img.copy()

            # Declaration of the filename of the annotations which is being treated
            annot_name = dir_annot + "/" + dir_name.split("/")[-1] + "/" + str(video_title) + str(nb_filename).zfill(5) + ".json"

            # Prediction of the bounding box. If precised, the method gives to the
            # predict function, the previous predicted bounding box
            if model.use_time:
                BB = model.predict(file_name, a=a, b=b, precBB=BB)
            else:
                BB = model.predict(file_name, a=a, b=b, precBB=[])

            # Computation of the number of frames per second
            fps = 1 / (time.time() - t_ite + 10e-10)

            # Read the ground truth annotation
            BB_ground_truth = read_annotation(annot_name)
            
            if use_kalman is not None and found:
                middle_x, middle_y = kf.predict()

            if (debug):
                img = cv.imread(file_name, cv.IMREAD_COLOR)
                x_a, y_a, w_a, h_a = BB_ground_truth
                cv.rectangle(img, (x_a, y_a), (x_a + w_a, y_a + h_a), (255, 0, 0), 2)
                cv.line(img, (0, int(b)), (640, int(a*640+b)), (0, 0, 255), 3)


            # If there is no bounded box found, the IoU is equal to O
            if BB == []:
                if use_kalman is None or not found:
                    IoU_video.append(0)
                    score_video.append(-1)
                    no_box += 1
            # Otherwise, the method compute the IoU
            else:

                x, y, w, h = BB

                if margin != 0:
                    x -= margin
                    y -= margin
                    h += 2*margin
                    w += 2*margin
                    BB = [x, y, w, h]

                if use_kalman is None or not found:
                    cropped_img = img_ini[y:y+h, x:x+w]
                    IoU_video.append(IoU(BB, BB_ground_truth))
                    score_video.append(score(BB_ground_truth, BB))

                if (use_kalman is None or not found) and IoU_video[-1] == 0:
                    no_box += 1

                if use_kalman is not None:
                    # Incrementation of the nb of value for the exponentially weighted average
                    t = t + 1

                    # Update the max or the weighted average for h and w
                    if use_kalman == "avg":
                        S_h = beta * S_h + (1-beta) * h
                        S_w = beta * S_w + (1-beta) * w
                        S_h_adj = S_h / (1 - beta**t)
                        S_w_adj = S_w / (1 - beta ** t)
                    elif use_kalman == "max":
                        if len(max_h)>=20:
                            max_h.pop(0)
                            max_w.pop(0)
                        max_h.append(h)
                        max_w.append(w)

                    # If a box is found, update the Kalman filter
                    if found:
                        middle_x, middle_y = kf.update([[x + w/2], [y + h/2]])


                    # Initialize Kalman filter
                    if not found:
                        found = True
                        x_ini = np.array([[x + w/2], [y + h/2], [0], [0]])
                        kf = KalmanFilter(1/60, x_ini)

                if debug:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(img, "Predicted BB", (x, y + h + 20), 0, 0.5, (0, 255, 0), 2)

            if use_kalman is not None and middle_x !=0 and middle_y != 0:
                if use_kalman == "avg":
                    x = int(middle_x - S_w_adj/2)
                    y = int(middle_y - S_h_adj/2)
                    w = int(S_w_adj)
                    h = int(S_h_adj)
                elif use_kalman == "max":
                    w = int(max(max_w))
                    h = int(max(max_h))
                    x = int(middle_x - w/2)
                    y = int(middle_y - h/2)
                    

                BB = [x, y, w, h]
                IoU_video.append(IoU(BB, BB_ground_truth))
                score_video.append(score(BB_ground_truth, BB))

                if IoU_video[-1] == 0:
                    no_box += 1

                if debug:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(img, "Kalman Filter", (x, y + h + 20), 0, 0.5, (0, 0, 255), 2)
                    cropped_img = img_ini[y:y+h, x:x+w]

            if debug:
                cv.putText(img, "IoU : " + str(round(IoU_video[-1], 2)), (10, 40), 0, 1.5, (0, 0, 255), 2)
                cv.putText(img, "Score : " + str(round(score_video[-1], 2)), (10, 450), 0, 1.5, (0, 0, 255), 2)
                cv.putText(img, "True BB", (x_a, y_a - 10), 0, 0.5, (255, 0, 0), 2)
                cv.putText(img, "FPS : " + str(round(fps, 2)), (350, 40), 0, 1.5, (0, 255, 255), 2)
                cv.imshow("Bounding box", img)
                cropped_img = cv.resize(cropped_img, (256, 256))
                cv.imshow("Cropped Image", cropped_img)
                if cv.waitKey(2) & 0xFF == ord('q'):
                    quit = True
                    break

            # Time computation for the computation of the FPS
            t_ite = time.time()

        # Put the values into a list and compute some results to display
        IoU_values.append(IoU_video)
        score_values.append(score_video)
        stat_values.append([np.sum(IoU_video) / nb, no_box, np.sum(IoU_video) / (nb - no_box), np.sum(score_video) / nb])
        stat_total[0] = stat_total[0] + np.sum(IoU_video)
        stat_total[1] = stat_total[1] + no_box
        stat_total[2] = stat_total[2] + np.sum(IoU_video)
        stat_total[3] = stat_total[3] + np.sum(score_video)
        nb_total += nb

        if debug:
            print("mean IoU : ", stat_values[-1][0], " Number of images with no box : ", stat_values[-1][1],
                " IoU mean with no box : ", stat_values[-1][2], " mean score : ", stat_values[-1][3])

        if quit:
            break

    stat_total[0] = stat_total[0]/nb_total
    stat_total[3] = stat_total[3]/nb_total
    stat_total[2] = stat_total[2]/(nb_total - stat_total[1])
    stat_values.append(stat_total)
    cv.destroyAllWindows()
    if debug:
            print("TOTAL mean IoU : ", stat_total[0], " Number of images with no box : ", stat_total[1],
                " IoU mean with no box : ", stat_total[2], " mean score : ", stat_total[3])

    return IoU_values, score_values, stat_values



