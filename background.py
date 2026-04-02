# features/background.py
from ultralytics import YOLO
import numpy as np

_model = None
_model_names = None

def init(config: dict):
    """
    config: {'weights': 'yolov8n.pt', 'device': 'cpu' or 'cuda:0'}
    """
    global _model, _model_names
    weights = config.get('weights', 'yolov8n.pt')
    device = config.get('device', None)
    if device:
        _model = YOLO(weights, device=device)
    else:
        _model = YOLO(weights)
    _model_names = _model.names if hasattr(_model, 'names') else {}

def run(frame=None, timestamp=None, **kwargs):
    """
    Run person detection on a single frame.
    Returns:
      {'persons': int, 'person_conf_max': float}
    """
    global _model, _model_names
    try:
        if frame is None:
            return {'persons': 0, 'person_conf_max': 0.0}
        results = _model(frame, stream=False)  # non-stream inference for single frame
        persons = 0
        max_conf = 0.0
        # results can be a list-like of Result objects
        for res in results:
            boxes = getattr(res, 'boxes', [])
            for b in boxes:
                cls_id = int(b.cls[0]) if hasattr(b, 'cls') else None
                conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                label = _model_names.get(cls_id, str(cls_id)) if _model_names else str(cls_id)
                if label.lower() == 'person' or label.lower() == 'person'[:]:  # safe check
                    persons += 1
                    if conf > max_conf: max_conf = conf
        return {'persons': int(persons), 'person_conf_max': float(max_conf)}
    except Exception as e:
        return {'persons': 0, 'person_conf_max': 0.0, 'background_error': str(e)}

