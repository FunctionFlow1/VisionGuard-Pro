import unittest
import cv2
import numpy as np
import os
from src.detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create dummy files for testing
        cls.dummy_model_path = "dummy_yolov3.weights"
        cls.dummy_config_path = "dummy_yolov3.cfg"
        cls.dummy_classes_path = "dummy_coco.names"

        with open(cls.dummy_model_path, "w") as f: f.write("dummy weights")
        with open(cls.dummy_config_path, "w") as f: f.write("dummy config")
        with open(cls.dummy_classes_path, "w") as f: f.write("person
car
bike")

    @classmethod
    def tearDownClass(cls):
        # Clean up dummy files
        os.remove(cls.dummy_model_path)
        os.remove(cls.dummy_config_path)
        os.remove(cls.dummy_classes_path)

    def test_detector_initialization(self):
        detector = ObjectDetector(self.dummy_model_path, self.dummy_config_path, self.dummy_classes_path)
        self.assertIsNotNone(detector.net)
        self.assertGreater(len(detector.classes), 0)

    def test_detect_method_on_dummy_frame(self):
        detector = ObjectDetector(self.dummy_model_path, self.dummy_config_path, self.dummy_classes_path)
        dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame) # Should return empty list as no objects are in dummy frame
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 0)

    def test_draw_boxes_method(self):
        detector = ObjectDetector(self.dummy_model_path, self.dummy_config_path, self.dummy_classes_path)
        dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
        detections = [
            {"box": (10, 10, 50, 50), "label": "person", "confidence": "0.95"}
        ]
        annotated_frame = detector.draw_boxes(dummy_frame.copy(), detections)
        # Basic check: ensure the frame is still an image and not empty
        self.assertIsInstance(annotated_frame, np.ndarray)
        self.assertFalse(np.array_equal(dummy_frame, annotated_frame)) # Should be different after drawing

if __name__ == '__main__':
    unittest.main()
