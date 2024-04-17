import logging

from ultralytics import YOLO
import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter


logger = logging.getLogger(__name__)

DETECTOR_KEY = "yolo"


class YoloDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]


class YoloDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: YoloDetectorConfig):
        self.config = detector_config
        self.model = YOLO(self.config.model.path, task='detect')

    def detect_raw(self, tensor_input):
        indata = np.squeeze(np.ascontiguousarray(tensor_input))

        output = self.model(indata, verbose=False, conf=0.01)

        if len(output) != 1:
            return
        size = output[0].orig_shape[0]
        results = output[0].boxes
        count = len(results.cls)
        class_ids = results.cls.int().tolist()
        boxes = [[c/size for c in b] for b in results.xyxy.tolist()]
        scores = [x**self.config.model.confidence_gamma for x in results.conf.tolist()]
        detections = np.zeros((20, 6), np.float32)

        for i in range(count):
            if i == 20:
                break
            detections[i] = [
                class_ids[i],
                float(scores[i]),
                boxes[i][1],
                boxes[i][0],
                boxes[i][3],
                boxes[i][2],
            ]

        return detections
