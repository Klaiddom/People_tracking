from typing import List

from detector import PoseEstiamtion
from tracking import Tracker
import cv2
import numpy as np
import json


def prepare_data_for_tracker(out: List[np.ndarray]) -> List[np.ndarray]:
    points = []
    for person in out:
        if person[2][0] < 1 and person[5][0] > 0:
            chest = np.array(person[5])[:-1]
            points.append(chest)
        elif person[2][0] > 0 and person[5][0] < 1:
            chest = np.array(person[2])[:-1]
            points.append(chest)
        elif person[2][0] > 0 and person[5][0] > 0:
            chest = np.mean([person[2], person[5]], axis=0)[:-1]
            points.append(chest)
    return points


def draw(frame: np.ndarray, out: List[np.ndarray]) -> None:
    for person in out:
        for point in person:
            frame = cv2.circle(frame, (int(point[0]), int(point[1])), radius, (0, 0, 255), thickness, cv2.FILLED)
    for person_id, point in tracked_dict.items():
        frame = cv2.putText(frame, f'person_{person_id}', (int(point[0]), int(point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_4)


if __name__ == '__main__':
    is_use_hashed_data = True
    video_streams = cv2.VideoCapture('pedastrians.mp4')
    tracker = Tracker()
    color = (255, 0, 0)
    radius = 3
    thickness = 2
    counter = 0
    if is_use_hashed_data:
        with open('core/skeletons.json', 'r') as file:
            frame2persons = json.load(file)
    else:
        pose_estimator = PoseEstiamtion(prediction_size=(640, 360), init_size=(1280, 720))

    while True:
        _, frame = video_streams.read()
        if frame is None:
            break

        if is_use_hashed_data:
            out = frame2persons[str(counter)]
        else:
            out = pose_estimator.predict(frame)

        points = prepare_data_for_tracker(out)
        tracked_dict = tracker.track(points, str(counter))
        draw(frame, out)
        cv2.imshow(f'frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        counter += 1