import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, config_path, classes_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame, confidence_threshold=0.5, nms_threshold=0.4):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        results = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                results.append({
                    "box": (x, y, w, h),
                    "label": label,
                    "confidence": confidence
                })
        return results

    def draw_boxes(self, frame, detections):
        for det in detections:
            x, y, w, h = det["box"]
            label = det["label"]
            confidence = det["confidence"]
            color = (0, 255, 0) # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

if __name__ == "__main__":
    # This is a placeholder for actual model/config/classes files
    # In a real scenario, these would be downloaded or provided.
    # For demonstration, we'll create dummy files.
    dummy_model_path = "dummy_yolov3.weights"
    dummy_config_path = "dummy_yolov3.cfg"
    dummy_classes_path = "dummy_coco.names"

    with open(dummy_model_path, "w") as f: f.write("dummy weights")
    with open(dummy_config_path, "w") as f: f.write("dummy config")
    with open(dummy_classes_path, "w") as f: f.write("person
car
bike")

    try:
        detector = ObjectDetector(dummy_model_path, dummy_config_path, dummy_classes_path)
        dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
        # Simulate a detection
        # detections = detector.detect(dummy_frame)
        # annotated_frame = detector.draw_boxes(dummy_frame, detections)
        print("ObjectDetector initialized successfully (using dummy files).")
    except Exception as e:
        print(f"Error initializing ObjectDetector: {e}")
    finally:
        os.remove(dummy_model_path)
        os.remove(dummy_config_path)
        os.remove(dummy_classes_path)
