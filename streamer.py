import cv2
import os


class Streamer:

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise Exception(f'File {file_path} does not exists.')
        self.stream = cv2.VideoCapture(file_path)