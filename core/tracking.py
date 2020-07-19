import numpy as np
import json
from typing import List, Dict, Tuple


class Tracker():

    def __init__(self, time_back=5):
        self.history: Dict[str, Dict[str, np.ndarray]] = dict()
        self.movement_vectors: Dict[str, np.ndarray] = dict()
        self.time_to_look_back = time_back
        self.ghost_persons = []
        self.stopped_persons = dict()
        self.next_persons_ids = 0

    def track(self, person_points: List[np.ndarray], frame_number: str) -> Dict[str, np.ndarray]:
        """Main method, perform all calculations here, points in person points should be poped"""
        matched_dict = dict()
        for person_id in self.history.keys():
            if person_id in self.ghost_persons:
                continue
            nearest_set_of_points = self.find_nearest_points(person_id, person_points)
            # matched_point, score = self.cosine_similarity(nearest_set_of_points, person_id, frame_number)
            if nearest_set_of_points:
                matched_point = nearest_set_of_points[0]
            else:
                matched_point = np.array([])
            if matched_point.any():
                if person_id not in self.history:
                    self.history[person_id] = {}
                self.history[person_id][frame_number] = matched_point
                self.movement_vectors = self.calculate_movement_vector({str(person_id): self.history[person_id][frame_number]},
                                                                                  str(frame_number))
                matched_dict[person_id] = matched_point
        person_points = self.pop_matched_points(person_points, list(matched_dict.values()))
        for point in person_points:
            self.history[str(self.next_persons_ids)] = {}
            self.history[str(self.next_persons_ids)][frame_number] = [point]
            matched_dict[str(self.next_persons_ids)] = point
            self.next_persons_ids += 1
        return matched_dict

    def calculate_movement_vector(self, current_states: Dict[str, np.ndarray], frame_id: str) -> Dict[str, np.ndarray]:
        """From matched persons to the previous frame calculate movment vector"""
        movement_vectors = dict()
        for person_id in current_states:
            if str(int(frame_id) - 1) in self.history[person_id]:
                movement_vectors[person_id] = current_states[person_id] - \
                                            self.history[person_id][str(int(frame_id) - 1)]
        return movement_vectors


    #unused
    #shit code
    def cosine_similarity(self, undefined_points: List[np.ndarray], target_id: str, frame_id: str, threshold=0.95) \
        -> Tuple[np.ndarray, float]:
        """Calculate cosine distance between vectors and returns more similar point and score"""
        cosines = []
        if target_id in self.movement_vectors:
            if undefined_points is None:
                print ()
            for point in undefined_points:
                try:
                    new_vec = point - self.history[target_id][str(int(frame_id) - 1)]
                except:
                    cosines.append(-2)
                    continue
                production = self.movement_vectors[target_id].dot(np.transpose(new_vec))
                norms = np.linalg.norm(self.movement_vectors[target_id]) * np.linalg.norm(new_vec)
                cosine = production / norms
                if np.linalg.norm(new_vec) < 25:
                    cosines.append(cosine)
                else:
                    cosines.append(-2)
            if cosines:
                if np.max(cosines) < threshold:
                    return np.array([]), -1.
                else:
                    return undefined_points[cosines.index(max(cosines))], max(cosines)
            else:
                return np.array([]), 0
        if undefined_points:
            return undefined_points[-1], 1
        else:
            return np.array(undefined_points), 0

    def find_nearest_points(self, target_id: str, all_current_points: List[np.ndarray], dist_threshold=50, k=3) -> \
            List[np.ndarray]:
        """calculate nearset points to the target point from previous frame,
        dist threshold should be found in pixels"""
        distance = []
        for point in all_current_points:
            dist = np.linalg.norm(point - self.history[target_id][max(k for k, v in self.history[target_id].items())])
            distance.append(dist)
        sorted_points = [point for dist, point in sorted(zip(distance, all_current_points))][:k]
        distance = sorted(distance)[:k]
        if distance:
            for dist in distance:
                if dist > dist_threshold:
                    return sorted_points[:distance.index(dist)]
                else:
                    return sorted_points
        return []

    def is_person_stopped(self, stops_threshold=10) -> bool:
        """returns True if distance less than stops thresh, False otherwise"""
        for person_id in self.movement_vectors:
            if np.linalg.norm(self.movement_vectors[person_id][0] - self.movement_vectors[person_id][1]) \
                    < stops_threshold:
                self.stopped_persons[person_id] = \
                    self.history[person_id][max(k for k, v in self.history[person_id].items())]
                return True
            else:
                return False

    def pop_matched_points(self, original_list: List[np.ndarray], matched_points: List[np.ndarray]) \
            -> List[np.ndarray]:
        """pops from original list matched points"""
        new_list = []
        poped_list = []
        for point in original_list:
            if not np.in1d(point, matched_points).all():
                new_list.append(point)
            else:
                poped_list.append(point)
        if len(new_list) == 0 and not self.history and \
                len(poped_list) == len(matched_points) and len(original_list) < 2:
            new_list = original_list
        return new_list


if __name__ == "__main__":
    #Check tracker worck from hashed data
    first_track = Tracker()
    with open('skeletons.json', 'r') as file:
        people_all_frames = json.load(file)
    for frame_id, persons in people_all_frames.items():
        points = []
        for person in persons:
            chest = np.mean([person[2], person[5]], axis=0)[:-1]
            if (chest > 1).all():
                points.append(chest)
        first_track.track(points, frame_id)
    print ('End')
