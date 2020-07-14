from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import os
import numpy as np


class Streamer:
    """
    Video preprocessing.
    """
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise Exception(f'File {file_path} does not exists.')
        self.file_path = file_path
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
        # cv2.destroyWindow('frame')

    def cut_video(self, start_time, end_time, new_name):
        """
        Cut a video staying just important part. Save new video named new_name.
        :param start_time: float
        :param end_time: float
        :param new_name: str
        :return:
        """
        ffmpeg_extract_subclip(self.file_path, start_time, end_time, new_name)


if __name__=="__main__":
    stream = Streamer('../Video/Ground Collision at LAX- United Airlines vs. Air Canada.mp4')
    stream.cut_video(0, 50, '../Video/test_video.mp4')
    first_frame = stream.get_frame()
    stream.display(first_frame)
