# features/gaze.py
import cv2
import mediapipe as mp
import numpy as np
import collections
import time

_fm = None
_baseline_ratio = None
_baseline_vert = None
_calib_samples = []
_CALIB_N = 15  # number of frames used to calibrate baseline
_deque_r = collections.deque(maxlen=5)
_deque_v = collections.deque(maxlen=5)
_flag_time = 1.5
_last_flag_time = None
_flag_count = 0

# indices from MediaPipe face mesh used by original code
LEFT_EYE, RIGHT_EYE = [33,133], [362,263]
LEFT_IRIS, RIGHT_IRIS = [474,475,476,477], [469,470,471,472]

def init(config: dict):
    """
    Initialize FaceMesh and reset calibration.
    """
    global _fm, _baseline_ratio, _baseline_vert, _calib_samples, _deque_r, _deque_v, _flag_count, _last_flag_time
    _fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)
    _baseline_ratio = None
    _baseline_vert = None
    _calib_samples = []
    _deque_r = collections.deque(maxlen=5)
    _deque_v = collections.deque(maxlen=5)
    _flag_count = 0
    _last_flag_time = None

def _get_gaze_metrics(mesh, w, h):
    le, re = mesh[LEFT_EYE], mesh[RIGHT_EYE]
    li, ri = mesh[LEFT_IRIS], mesh[RIGHT_IRIS]
    lc, rc = np.mean(li, 0), np.mean(ri, 0)
    lr = (lc[0]-le[0,0])/(le[1,0]-le[0,0]+1e-6)
    rr = (rc[0]-re[0,0])/(re[1,0]-re[0,0]+1e-6)
    gaze_ratio = (lr+rr)/2
    avg_iris_y = (lc[1]+rc[1])/2
    avg_eye_y = (le[0,1]+le[1,1]+re[0,1]+re[1,1])/4
    vert_diff = avg_iris_y - avg_eye_y
    return gaze_ratio, vert_diff

def run(frame=None, timestamp=None, **kwargs):
    """
    Process one frame. If baseline not set, accumulates calibration frames.
    Returns dict:
      {'gaze_state': 'Forward'/'Left'/'Right'/'Up'/'Down', 'gaze_flag':0/1, 'gaze_flags_total': int, 'calibrating':0/1}
    """
    global _fm, _baseline_ratio, _baseline_vert, _calib_samples, _deque_r, _deque_v, _flag_count, _last_flag_time
    try:
        if frame is None:
            return {'gaze_state': 'no_frame', 'gaze_flag': 0, 'gaze_flags_total': _flag_count, 'calibrating': 0}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _fm.process(rgb)
        if not res.multi_face_landmarks:
            return {'gaze_state': 'no_face', 'gaze_flag': 0, 'gaze_flags_total': _flag_count, 'calibrating': int(_baseline_ratio is None)}

        mesh = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark])
        r, v = _get_gaze_metrics(mesh, w, h)

        # calibration phase
        if _baseline_ratio is None:
            _calib_samples.append((r, v))
            if len(_calib_samples) >= _CALIB_N:
                vals = np.array(_calib_samples)
                _baseline_ratio = float(np.mean(vals[:, 0]))
                _baseline_vert = float(np.mean(vals[:, 1]))
                _calib_samples = []
                return {'gaze_state': 'calibrated', 'gaze_flag': 0, 'gaze_flags_total': _flag_count, 'calibrating': 0}
            else:
                return {'gaze_state': 'calibrating', 'gaze_flag': 0, 'gaze_flags_total': _flag_count, 'calibrating': 1, 'calib_progress': len(_calib_samples)}

        # normal operation
        _deque_r.append(r); _deque_v.append(v)
        r_mean, v_mean = float(np.mean(_deque_r)), float(np.mean(_deque_v))
        dx = r_mean - _baseline_ratio
        dy = v_mean - _baseline_vert

        # thresholds (same semantics as your original code)
        H_THRESH = 0.05
        V_THRESH = 4.0

        direction = "Forward"
        if dx < -H_THRESH:
            direction = "Left"
        elif dx > H_THRESH:
            direction = "Right"
        elif dy < -V_THRESH:
            direction = "Up"
        elif dy > V_THRESH:
            direction = "Down"

        flag = 0
        if direction != "Forward":
            if _last_flag_time is None:
                _last_flag_time = timestamp or time.time()
            else:
                if (timestamp or time.time()) - _last_flag_time >= _flag_time:
                    _flag_count += 1
                    flag = 1
                    _last_flag_time = (timestamp or time.time())
        else:
            _last_flag_time = None

        return {'gaze_state': direction, 'gaze_flag': int(flag), 'gaze_flags_total': int(_flag_count), 'calibrating': 0}

    except Exception as e:
        return {'gaze_state': 'error', 'gaze_flag': 0, 'gaze_flags_total': _flag_count, 'gaze_error': str(e)}

