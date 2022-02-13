import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/test_video.mp4"

# # Replace these with the directories containing your
# # positive and negative sample images, respectively.
# neg_dir = "D:/code/ProjectOfPyCharm/Lab/Machine_Learning/Lab07/samples/non-vehicles"
# pos_dir = "D:/code/ProjectOfPyCharm/Lab/Machine_Learning/Lab07/samples/vehicles"
#
# # Replace this with the path to your test video file.
# video_file = "D:/code/ProjectOfPyCharm/Lab/Machine_Learning/Lab07/videos/test_video.mp4"

def experiment1():
    for ele in {"gray", "hls", "hsv", "lab", "luv", "ycrcb", "yuv"}:
        feature_data = processFiles(pos_dir, neg_dir, recurse=True,
                                hog_features=True,hist_features=True,spatial_features=False,color_space=ele)
        classifier_data = trainSVM(feature_data=feature_data)


def experiment2():
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
                                hog_features=True,hist_features=True,spatial_features=False)

    classifier_data = trainSVM(feature_data=feature_data)
    detector = Detector(init_size=(64,64), x_overlap=0.55, y_step=0.008,
            x_range=(0.66, 0.88), y_range=(0.52, 0.67), scale=1.5)
    detector.loadClassifier(classifier_data=classifier_data)
    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap,write=True)
    print('video has saved')

if __name__ == "__main__":
    # experiment1()
    experiment2()
