from typing import List
import cv2
from PIL import Image
import numpy as np
import openpifpaf
import torch


class PoseEstiamtion():

    def __init__(self, checkpoint='shufflenetv2k16w', device=torch.device('cuda'),
                 init_size=(1920, 1080), prediction_size=(480, 270)):
        self.model, _ = openpifpaf.network.factory(checkpoint=checkpoint)
        self.model = self.model.to(device)
        openpifpaf.decoder.CifSeeds.threshold = 0.4
        openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.1
        openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.1
        self.processor = openpifpaf.decoder.factory_decode(self.model.head_nets,
                                                           basenet_stride=self.model.base_net.stride)
        self.preprocesser = openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(16),
            openpifpaf.transforms.EVAL_TRANSFORM
        ])
        self.device = device
        self.input_shape = prediction_size
        self.initial_size = init_size
        self.scale_factor = self.initial_size[0] / self.input_shape[0]

    def transform_frame(self, frame:np.ndarray) -> Image.Image:
        transformed_frame = Image.fromarray(cv2.resize(frame, self.input_shape))
        return transformed_frame

    def get_loader(self, frame: Image.Image) -> torch.utils.data.DataLoader:
        img = openpifpaf.datasets.PilImageList([frame], preprocess=self.preprocesser)
        data_loader = torch.utils.data.DataLoader(
            img, batch_size=1, pin_memory=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        return data_loader

    def extract_kp(self, predictions: List) -> List[List[np.ndarray]]:
        result = []
        for pifpaf_prediciton in predictions:
            for person_prediction in pifpaf_prediciton:
                unscaled_kp: np.ndarray = person_prediction.data
                result.append(unscaled_kp * self.scale_factor)
        return result

    def predict(self, frame: np.ndarray) -> List[List[np.ndarray]]:
        transformed_frame = self.transform_frame(frame)
        loader = self.get_loader(transformed_frame)
        predictions = []
        for img_batch, _, _ in loader:
            predictions = self.processor.batch(self.model, img_batch, device=self.device)
        predictions = self.extract_kp(predictions)
        return predictions


if __name__ == '__main__':
    video_streams = cv2.VideoCapture('C:\\Users\\User\\Downloads\\pedastrians.mp4')
    pose_estimator = PoseEstiamtion(prediction_size=(640, 360), init_size=(1280, 720))
    kp_color = (255, 0, 0)
    radius = 3
    thickness = 2
    visualize_output = True
    while True:
        _, frame = video_streams.read()
        if frame is None:
            break
        out = pose_estimator.predict(frame)
        if visualize_output:
            for pedastrians in out:
                if pedastrians:
                    for person_kp in pedastrians:
                        for point in person_kp:
                            frame = cv2.circle(frame, (int(point[0]), int(point[1])),
                                               radius, kp_color, thickness, cv2.FILLED)
        cv2.imshow(f'frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

