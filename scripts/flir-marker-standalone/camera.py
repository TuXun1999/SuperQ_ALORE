"""Aravis capture adapted from the RPM focus-viewer camera implementation."""
import json
import sys
import numpy as np

SERIAL="19476401"

def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)

def unpack_bayer(data, width, height):
    """Reject truncated payloads even when transport reports SUCCESS."""
    if len(data) != width * height:
        return None
    return np.frombuffer(data, np.uint8).reshape(height, width).copy()

def import_aravis():
    # RPM's venv and system GI both use Python 3.12. Append, never override the
    # locked venv's NumPy/OpenCV with system packages.
    sys.path.append("/usr/lib/python3/dist-packages")
    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    return Aravis

class Camera:
    features = {
        "string": ["ExposureAuto", "GainAuto", "TriggerMode", "AcquisitionMode", "PixelFormat"],
        "float": ["ExposureTime", "Gain", "AcquisitionFrameRate"],
        "boolean": ["IspEnable", "AcquisitionFrameRateEnable"],
    }

    def __init__(self, args, out):
        self.args, self.out = args, out
        self.c = self.stream = self.original = None
        self.good = self.bad = self.discarded = self.timeouts = 0
        self.bad_payloads = 0
        self.last_frame_id = None
        self.A = import_aravis()

    def read_settings(self):
        settings = {key: getattr(self.c, "get_" + typ)(key)
                    for typ, keys in self.features.items() for key in keys}
        settings["region"] = list(self.c.get_region())
        return settings

    def open(self):
        c = self.c = self.A.Camera.new(self.args.camera)
        if c.get_device_serial_number() != SERIAL:
            raise RuntimeError("Unexpected camera serial; this launcher targets only " + SERIAL)
        if c.get_model_name() != "Blackfly S BFS-PGE-50S5C":
            raise RuntimeError("Unexpected camera model")
        if not c.get_device().is_controller():
            raise RuntimeError("Camera is owned by another process; no takeover attempted")
        for name in ["BinningHorizontal", "BinningVertical", "DecimationHorizontal", "DecimationVertical"]:
            if c.get_integer(name) != 1:
                raise RuntimeError(name + " must be 1 for calibrated native-pixel poses")
        if c.get_string("TriggerSelector") != "FrameStart":
            raise RuntimeError("TriggerSelector must already be FrameStart")
        for key in ["ReverseX", "ReverseY"]:
            if c.get_boolean(key):
                raise RuntimeError(key + " must be disabled for this calibration")
        self.original = self.read_settings()
        atomic_json(self.out / "original-settings.json", self.original)
        c.stop_acquisition()
        c.set_string("TriggerMode", "Off")
        c.set_string("AcquisitionMode", "Continuous")
        c.set_string("ExposureAuto", "Off")
        c.set_string("GainAuto", "Off")
        c.set_exposure_time(self.args.exposure_us)
        requested_gain = getattr(self.args, "gain_db", None)
        if requested_gain is not None:
            low, high = c.get_gain_bounds()
            if not low <= requested_gain <= high:
                raise ValueError(f"Requested gain outside camera bounds {low}..{high} dB")
            c.set_gain(requested_gain)
        c.set_boolean("IspEnable", False)
        c.set_string("PixelFormat", "BayerRG8")
        c.set_region(0, 0, 2448, 2048)
        c.set_boolean("AcquisitionFrameRateEnable", self.args.fps > 0)
        if self.args.fps > 0:
            c.set_frame_rate(self.args.fps)
        self.exposure = c.get_exposure_time()
        self.gain = c.get_gain()
        self.temperature = c.get_float("DeviceTemperature")
        atomic_json(self.out / "viewer-settings.json", self.read_settings())
        self.stream = c.create_stream(None, None)
        for _ in range(8):
            self.stream.push_buffer(self.A.Buffer.new_allocate(c.get_payload()))
        if not np.isfinite(self.temperature) or self.temperature >= 70:
            raise RuntimeError("Camera already at or above 70 C")
        c.start_acquisition()

    def read(self):
        b = self.stream.timeout_pop_buffer(200000)
        if b is None:
            self.timeouts += 1
            return None
        # Keep the freshest available buffer; never accumulate a display queue.
        while True:
            newer = self.stream.try_pop_buffer()
            if newer is None:
                break
            if b.get_status() != self.A.BufferStatus.SUCCESS:
                self.bad += 1
            self.stream.push_buffer(b)
            self.discarded += 1
            b = newer
        try:
            if b.get_status() != self.A.BufferStatus.SUCCESS:
                self.bad += 1
                return None
            h, w = b.get_image_height(), b.get_image_width()
            if (w, h) != (2448, 2048):
                raise RuntimeError(f"Unexpected image shape {(w, h)}")
            raw = unpack_bayer(b.get_data(), w, h)
            if raw is None:
                self.bad += 1
                self.bad_payloads += 1
                return None
            self.last_frame_id = int(b.get_frame_id())
            self.good += 1
            return raw
        finally:
            self.stream.push_buffer(b)

    def adjust_exposure(self, multiplier):
        lo, hi = self.c.get_exposure_time_bounds()
        self.c.set_exposure_time(min(hi, 100000, max(lo, self.exposure * multiplier)))
        self.exposure = self.c.get_exposure_time()

    def close(self):
        if self.c is None or self.original is None:
            return []
        c, d, errors = self.c, self.original, []
        def step(label, fn):
            try:
                fn()
            except Exception as e:
                errors.append(f"{label}: {e}")
        step("stop acquisition", c.stop_acquisition)
        self.stream = None
        for key in ["ExposureAuto", "GainAuto"]:
            step(key, lambda key=key: c.set_string(key, "Off"))
        step("region", lambda: c.set_region(*d["region"]))
        for key in ["ExposureTime", "Gain"]:
            step(key, lambda key=key: c.set_float(key, d[key]))
        step("IspEnable", lambda: c.set_boolean("IspEnable", d["IspEnable"]))
        step("PixelFormat", lambda: c.set_string("PixelFormat", d["PixelFormat"]))
        # Restore the hidden manual rate even if the original limiter was off.
        step("rate enable", lambda: c.set_boolean("AcquisitionFrameRateEnable", True))
        step("AcquisitionFrameRate", lambda: c.set_float("AcquisitionFrameRate", d["AcquisitionFrameRate"]))
        step("rate enable restore", lambda: c.set_boolean("AcquisitionFrameRateEnable", d["AcquisitionFrameRateEnable"]))
        for key in ["AcquisitionMode", "TriggerMode", "GainAuto", "ExposureAuto"]:
            step(key, lambda key=key: c.set_string(key, d[key]))
        step("readback", lambda: atomic_json(self.out / "restored-settings.json", self.read_settings()))
        try:
            restored = self.read_settings()
            if restored != d:
                errors.append("Restoration readback differs from original settings")
        except Exception as exc:
            errors.append("Restoration verification: " + str(exc))
        return errors
