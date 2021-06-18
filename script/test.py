from Code.ColorSegmentation import ColorBB
from Code.Metrics.ModelPerformance import IoU_video
from Code.RandomForest import RandomForestBB
from Code.Preprocessing.WaterSurfaceDetection import surface_detection

Color = ColorBB("hsv")

#a, b = surface_detection(f"../Images/Testset/background/T1T2", 117, adjust_pt1=0, adjust_pt2=0)

#Color.predict("../Images/Testset/T200380.jpg", debug=True, a=a, b=b)

IoU_video("Images/Testset", "Annotations", Color, debug=True)
