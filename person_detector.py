# person_detector.py
import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.35, iou=0.45, imgsz=480, max_det=6):
        self.model = YOLO(weights)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        # COCO class id 0 = "person"
        self.person_id = 0

    def detect(self, frame_bgr):
        res = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[self.person_id],
            max_det=self.max_det,
            verbose=False
        )[0]
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy
        conf = res.boxes.conf
        if hasattr(xyxy, "detach"): xyxy = xyxy.detach()
        if hasattr(conf, "detach"): conf = conf.detach()
        xyxy = xyxy.cpu().numpy()
        conf = conf.cpu().numpy()
        for box, cf in zip(xyxy, conf):
            x1, y1, x2, y2 = box.astype(int)
            out.append((x1, y1, x2, y2, float(cf)))
        # sort by confidence desc
        out.sort(key=lambda b: b[4], reverse=True)
        return out
