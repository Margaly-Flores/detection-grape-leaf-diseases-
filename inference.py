import torch
import numpy as np
#import cv2
#import time
from ultralytics import YOLO
from PIL import Image, ImageDraw
from numpy import asarray
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


weight_dir = "./weights"
torch.hub.set_dir(weight_dir)


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n\nDevice Used:", self.device)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        # model = torch.hub.load("ultralytics/yolov8", "yolov8s", pretrained=False)
        # model.load_state_dict(torch.load("./weights/best.pt"))
        # model.eval()
        model = YOLO("./weights/best.pt")
        # -----------------------------------------------------------------------------

        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo8 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model.predict(source=frame)
        labels, cord = [r[-1] for r in results[0].boxes.boxes], [
            r[0:5] for r in results[0].boxes.boxes
        ]

        # print("labels, cord", labels, cord)
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        #frame = Image.fromarray(frame)
        labels, cord = results
        n = len(labels)
        #x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                """x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )"""
                x1, y1, x2, y2 = (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                )
                bgr = (0, 255, 0)
                plt.imshow(frame)
                fig, ax = plt.subplots()
                ax.imshow(frame)
                rect = patches.Rectangle((x2, y2), (x1-x2), (y1-y2), linewidth=3, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                frame = Image.fromarray(frame)
                frame = frame.convert('RGB')
                fig.savefig('frame.JPEG')
                frame = Image.open('frame.JPEG')
                #cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # cv2.putText(
                #     frame,
                #     self.class_to_label(labels[i]),
                #     (x1, y1),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.9,
                #     bgr,
                #     2,
                # )
                lab = self.class_to_label(labels[i])

        return frame,lab 

    def __call__(self,image_file): 
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        #frame = cv2.imread("image/image_leafblight.JPG")
        frame = Image.open(image_file)                              ############
        #frame = cv2.cvtColor(numpy.array(frame), cv2.COLOR_RGB2BGR) 
        frame = asarray(frame)                                      
        results = self.score_frame(frame)
        frame,lab = self.plot_boxes(results, frame)
        # print(lab)
        #cv2.imshow("img", frame)  #comment
        #cv2.waitKey(0)            #comment

        # closing all open windows
        #cv2.destroyAllWindows()   #comment
        return frame,lab 
    

# Create a new object and execute. 
# detection = ObjectDetection()
# detection()


