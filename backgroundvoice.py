# features/backgroundvoice.py
"""
Background voice detector with streaming (Silero VAD + sounddevice).
Provides:
 - init(config)         : loads model and sets parameters
 - start(alert_callback): spawns audio input stream and thread; calls alert_callback(source, message) on detections
 - stop()               : stops streaming thread and closes sounddevice stream
 - run(frame=None,... ) : compatibility single-check mode (process an audio_chunk if passed)

Notes:
 - Uses sounddevice.InputStream for low-latency audio capture.
 - Uses torch.hub Silero VAD utilities for speech timestamps.
"""
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch
from datetime import datetime
import os
import platform

# module state
_silero_model = None
_get_speech_timestamps = None
_sample_rate = 16000
_frame_duration = 0.5   # block duration in seconds (adjustable)
_threshold = 0.3
_stream = None
_stream_thread = None
_stream_stop_evt = None
_audio_q = None
_alert_cb = None
_max_buffer_seconds = 2.0
_last_alert_time = 0.0
_alert_cooldown = 2.0   # seconds between repeated voice alerts

LOG_PATH = "logs/voice_log.txt"

def init(config: dict):
    """
    Load Silero VAD model. Optional config keys:
      - 'repo' : torch hub repo (default 'snakers4/silero-vad')
      - 'sample_rate' : 16000 default
      - 'frame_duration' : seconds per audio block (0.2-0.8)
      - 'threshold' : silero threshold
      - 'alert_cooldown' : seconds
    """
    global _silero_model, _get_speech_timestamps, _sample_rate, _frame_duration, _threshold, _alert_cooldown
    repo = config.get('repo', 'snakers4/silero-vad')
    print(f"[backgroundvoice] Loading Silero VAD from {repo} ...")
    model, utils = torch.hub.load(repo_or_dir=repo, model='silero_vad', force_reload=False)
    (_get_speech_timestamps, _, _, _, _) = utils
    _sample_rate = int(config.get('sample_rate', _sample_rate))
    _frame_duration = float(config.get('frame_duration', _frame_duration))
    _threshold = float(config.get('threshold', _threshold))
    _alert_cooldown = float(config.get('alert_cooldown', _alert_cooldown))
    # ensure logs directory exists
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    print("[backgroundvoice] Silero VAD loaded.")

def _process_audio_block(np_chunk):
    """
    Converts numpy audio chunk (float32, -1..1 or int16) to torch tensor
    and applies Silero VAD. Returns number of speech segments.
    """
    global _silero_model, _get_speech_timestamps, _sample_rate, _threshold
    try:
        import torch as _torch
        audio_tensor = _torch.from_numpy(np_chunk).float()
        timestamps = _get_speech_timestamps(audio_tensor, _silero_model, sampling_rate=_sample_rate, threshold=_threshold)
        return len(timestamps), timestamps
    except Exception as e:
        # on error, return zero segments
        return 0, []

def _audio_callback(indata, frames, time_info, status):
    """
    sounddevice callback — push mono float32 chunk to queue
    """
    global audio_q
    try:
        if status:
            # you can log status if needed
            pass
        # convert to mono if multichannel
        arr = indata.copy()
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        # ensure float32 in [-1,1]
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.float32)
        # push to queue (non-blocking)
        try:
            audio_q.put_nowait(arr)
        except queue.Full:
            # drop block if queue full
            pass
    except Exception:
        pass

def _stream_worker():
    """
    Worker thread: consumes audio blocks from queue, accumulates up to _max_buffer_seconds,
    runs VAD and triggers alert_callback when speech present.
    """
    global audio_q, _stream_stop_evt, _alert_cb, _last_alert_time, _max_buffer_seconds
    buffer = np.array([], dtype=np.float32)
    max_len = int(_sample_rate * _max_buffer_seconds)
    while not _stream_stop_evt.is_set():
        try:
            block = None
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                # allow periodic checks for stop event
                continue
            if block is None:
                continue
            # append and clip buffer length
            buffer = np.concatenate([buffer, block])
            if buffer.size > max_len:
                buffer = buffer[-max_len:]
            # Whenever we have at least one frame_duration worth of audio, inspect last chunk
            if buffer.size >= int(_sample_rate * _frame_duration):
                # take last frame_duration seconds
                chunk = buffer[-int(_sample_rate * _frame_duration):]
                segments_count, timestamps = _process_audio_block(chunk)
                if segments_count > 0:
                    now = time.time()
                    if now - _last_alert_time >= _alert_cooldown:
                        _last_alert_time = now
                        ts = datetime.now().strftime("%H:%M:%S")
                        msg = f"Human voice detected (segments={segments_count}) at {ts}"
                        # log
                        try:
                            with open(LOG_PATH, "a") as f:
                                f.write(f"{ts} | {segments_count} segments\n")
                        except Exception:
                            pass
                        if _alert_cb:
                            try:
                                _alert_cb('voice', msg)
                            except Exception:
                                pass
                # small sleep to avoid tight loop
                time.sleep(0.05)
        except Exception:
            # swallow worker exceptions so thread doesn't die silently
            time.sleep(0.1)
    # cleanup
    return

def start(alert_callback):
    """
    Start streaming audio capture in background.
    alert_callback(source, message) will be called on detection.
    """
    global _stream, _stream_thread, _stream_stop_evt, audio_q, _alert_cb
    if _stream_thread and _stream_thread.is_alive():
        print("[backgroundvoice] already running")
        return
    _alert_cb = alert_callback
    audio_q = queue.Queue(maxsize=16)
    _stream_stop_evt = threading.Event()
    # start worker thread
    _stream_thread = threading.Thread(target=_stream_worker, daemon=True)
    _stream_thread.start()
    # start sounddevice stream
    try:
        _stream = sd.InputStream(channels=1, samplerate=_sample_rate, blocksize = int(_sample_rate * _frame_duration),
                                 callback=_audio_callback)
        _stream.start()
        print("[backgroundvoice] audio stream started.")
    except Exception as e:
        # worker thread continues but stream failed
        print("[backgroundvoice] failed to start InputStream:", e)

def stop():
    """
    Stop streaming and worker thread.
    """
    global _stream, _stream_thread, _stream_stop_evt, audio_q
    try:
        if _stream:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass
            _stream = None
        if _stream_stop_evt:
            _stream_stop_evt.set()
        if _stream_thread:
            _stream_thread.join(timeout=1.0)
    except Exception:
        pass
    audio_q = None
    print("[backgroundvoice] stopped.")

def run(frame=None, timestamp=None, audio_chunk=None, **kwargs):
    """
    Compatibility function for orchestrator:
     - If audio_chunk (numpy array) is provided, will run VAD on it and return dict.
     - Otherwise returns zeros (no audio processed here).
    Returns: {'voice_detected':0/1, 'voice_segments':int}
    """
    try:
        if audio_chunk is None:
            return {'voice_detected': 0, 'voice_segments': 0}
        np_chunk = np.asarray(audio_chunk, dtype=np.float32)
        if np_chunk.ndim > 1:
            np_chunk = np_chunk.mean(axis=1)
        cnt, timestamps = _process_audio_block(np_chunk)
        return {'voice_detected': int(cnt > 0), 'voice_segments': int(cnt)}
    except Exception as e:
        return {'voice_detected': 0, 'voice_segments': 0, 'backgroundvoice_error': str(e)}

