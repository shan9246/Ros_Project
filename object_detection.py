import threading
import queue
import time
import numpy as np
import cv2
import torch
from controller import Camera


class HumanDetector:
    def __init__(self, camera: Camera, device='cpu', confidence_threshold=0.3):
        self.camera = camera
        self.width = camera.getWidth()
        self.height = camera.getHeight()
        self.device = device
        self.conf_threshold = confidence_threshold

        # Load YOLOv5s model trained on COCO (person = class 0)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
        self.model.conf = self.conf_threshold
        self.model.classes = [0]  # Only detect person

        self.frame_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self._detect_loop)
        self.thread.start()

    def _detect_loop(self):
        while not self.stop_event.is_set():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                results = self.model(frame)
                detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, class
                self.result_queue.put(detections)

    def detect_async(self):
        image = self.camera.getImage()
        if image:
            img_array = np.frombuffer(image, np.uint8).reshape((self.height, self.width, 4))
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
            self.frame_queue.put(img_rgb)

    def get_latest_detections(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return []

    def display_detections(self, detections):
        image = self.camera.getImage()
        if not image:
            return

        img_array = np.frombuffer(image, np.uint8).reshape((self.height, self.width, 4))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

        found_human = False

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f"Human {conf:.2f}"
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            found_human = True
            #print(f"Human detected at coordinates: x1={int(x1)}, y1={int(y1)}, x2={int(x2)}, y2={int(y2)}")

        #if found_human:
            #print("Human detected in the scene!")

        cv2.imshow("Human Detection", img_bgr)
        cv2.waitKey(1)


    def stop(self):
        self.stop_event.set()
        self.thread.join()
        cv2.destroyAllWindows()
