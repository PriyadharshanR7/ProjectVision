"""
YOLOv8n object-detection wrapper.

Returns a list of detection dicts with keys:
    bbox   – [x1, y1, x2, y2]  (pixel coords)
    conf   – float 0-1
    label  – COCO class name (str)
"""

from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONF, YOLO_IOU_NMS, YOLO_IMG_SIZE


class Detector:
    """Loads YOLOv8n once and exposes a simple `detect(frame)` method."""

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)

    def detect(self, frame):
        """Run inference on a BGR frame; return list[dict]."""
        results = self.model.predict(
            frame,
            conf=YOLO_CONF,
            iou=YOLO_IOU_NMS,
            imgsz=YOLO_IMG_SIZE,
            verbose=False,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(boxes.conf[i]),
                        "label": self.model.names[int(boxes.cls[i])],
                    }
                )
        return detections
