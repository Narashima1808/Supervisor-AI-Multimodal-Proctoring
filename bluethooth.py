# features/bluethooth.py
import subprocess
import json
import time

ALLOWED_DEVICE = None

def init(config: dict):
    """
    config: {'allowed_device': 'Device Name or MAC'}
    """
    global ALLOWED_DEVICE
    ALLOWED_DEVICE = config.get('allowed_device', config.get('allowed_device_name', ''))
    # keep empty allowed device if not provided

def _get_connected_devices_windows():
    """
    Return list of (name, instance_id) using PowerShell JSON output.
    Works on Windows only; returns [] on errors.
    """
    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-PnpDevice -Class Bluetooth | Where-Object { $_.Status -eq 'OK' } | ConvertTo-Json"
            ],
            capture_output=True, text=True, timeout=5
        )
        if not result.stdout.strip():
            return []
        devices = json.loads(result.stdout)
        if isinstance(devices, dict):
            devices = [devices]
        connected = []
        for d in devices:
            name = d.get('FriendlyName', '') or d.get('Name', '')
            inst = d.get('InstanceId', '')
            connected.append((name, inst))
        return connected
    except Exception:
        return []

def run(frame=None, timestamp=None, **kwargs):
    """
    Single-check run. Returns:
      {'bluetooth_count': int, 'unauthorized_found': 0/1}
    """
    try:
        devices = _get_connected_devices_windows()
        count = len(devices)
        unauth = 0
        if ALLOWED_DEVICE:
            # if any connected device is not the allowed one, set unauthorized flag
            for name, inst in devices:
                if ALLOWED_DEVICE.lower() not in (name or '').lower() and ALLOWED_DEVICE.lower() not in (inst or '').lower():
                    unauth = 1
                    break
        return {'bluetooth_count': int(count), 'bluetooth_unauth': int(unauth)}
    except Exception as e:
        return {'bluetooth_count': 0, 'bluetooth_unauth': 0, 'bluetooth_error': str(e)}

