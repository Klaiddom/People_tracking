import cv2
import os
import numpy as np



class Streamer:

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise Exception(f'File {file_path} does not exists.')
        self.stream = cv2.VideoCapture(file_path)
        # if transforms is not None:
        #     self.transforms = transforms

    def get_frame(self) -> np.ndarray:
        """
        Get a frame from tuple of boolean and frame if True else None
        :return: np.ndarray # image of frame
        """
        return self.stream.read()[1]

    def apply_frame_transform(self):
        pass

    def display(self, frame):
        """
        Displaying an image 'frame'. Waiting for any key and close the window.
        :param frame: ndarray
        :return:
        """
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyWindow('frame')
