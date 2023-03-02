import torch
from ultralytics import YOLO
from PIL import Image
from numpy import asarray
import numpy as np
import cv2

# import io


weight_dir = "./weights"
torch.hub.set_dir(weight_dir)


class ObjectDetection:
    """
    Class implements Yolo model to make inferences on a youtube video using OpenCV.
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
        model = YOLO("./weights/best.pt")
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
        labels_number, cord = results
        labels_text = []
        n = len(labels_number)
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                )
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                label_text = self.class_to_label(labels_number[i])
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )
                labels_text.append(label_text)

        return frame, labels_text

    def __call__(self, image_file):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        frame = Image.open(image_file)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frame = asarray(frame)
        results = self.score_frame(frame)
        frame, lab = self.plot_boxes(results, frame)
        return frame, lab


# Create a new object and execute.
# detection = ObjectDetection()
# detection()
