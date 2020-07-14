from ..preprocess.streamer import Streamer
import numpy as np
import cv2


class detector():
    """
    Human detection in 2d.
    """
    def __init__(self, detector=cv2.HOGDescriptor(), streamer=Streamer):
        self.detector = detector
        self.streamer = streamer

    def bboxes(self, frame, size):
        """
        Drow bounding boxes for detected objects.
        :param frame: np.ndarray
        :param size: tuple(int)
        :return:
        """
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        frame = cv2.resize(frame, size)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = self.detector.detectMultiScale(frame, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the picture
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
