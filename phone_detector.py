# phone_detector.py
import cv2
import torch
from ultralytics import YOLO

CELL_PHONE_ID = 67  # COCO id

class PhoneDetector:
    def __init__(self,
                 weights="yolov8n.pt",
                 imgsz_1=416,
                 conf_1=0.35,
                 imgsz_2=768,
                 conf_2=0.25,
                 iou=0.45,
                 second_pass_gate=0.50,
                 max_det=10):
        self.model = YOLO(weights)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.names = getattr(self.model, "names", None) or getattr(getattr(self.model, "model", None), "names", {})
        self.imgsz_1 = imgsz_1
        self.conf_1  = conf_1
        self.imgsz_2 = imgsz_2
        self.conf_2  = conf_2
        self.iou     = iou
        self.max_det = max_det
        self.second_pass_gate = second_pass_gate

    def _predict(self, frame_bgr, imgsz, conf):
        return self.model.predict(
            frame_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=self.iou,
            classes=[CELL_PHONE_ID],
            max_det=self.max_det,
            verbose=False
        )[0]

    def detect(self, frame_bgr):
        res1 = self._predict(frame_bgr, self.imgsz_1, self.conf_1)
        dets = self._to_list(res1)
        best = max((d[4] for d in dets), default=0.0)

        if best < self.second_pass_gate:
            res2 = self._predict(frame_bgr, self.imgsz_2, self.conf_2)
            dets2 = self._to_list(res2)
            if dets2 and (max(d[4] for d in dets2) >= best):
                return dets2
        return dets

    def _to_list(self, res):
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy
        conf = res.boxes.conf
        cls  = res.boxes.cls
        if hasattr(xyxy, "detach"): xyxy = xyxy.detach()
        if hasattr(conf, "detach"): conf = conf.detach()
        if hasattr(cls,  "detach"): cls  = cls.detach()
        xyxy = xyxy.cpu().numpy()
        conf = conf.cpu().numpy()
        cls  = cls.cpu().numpy()
        for box, c, cf in zip(xyxy, cls, conf):
            x1, y1, x2, y2 = box.astype(int)
            out.append((x1, y1, x2, y2, float(cf), int(c)))
        return out

    def draw(self, frame_bgr, detections):
        for (x1, y1, x2, y2, cf, cid) in detections:
            label = f"{self.names.get(cid, cid)} {cf:.2f}"
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 160, 255), 2)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 160, 255), -1)
            cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
