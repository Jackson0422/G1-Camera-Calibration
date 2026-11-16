"""Native DepthAI v2/v3. Run on the machine physically hosting OAK USB."""
import time
import numpy as np
from .base import Frame


class OakCamera:
    def __init__(self, config):
        import depthai as dai
        self.config, self.dai = config, dai
        self.device = self.pipeline = None
        self.started = False
        self.previous = None
        self.major = int(dai.__version__.split('.')[0])
        try:
            devices = dai.Device.getAllAvailableDevices()
            serial = config.get('expected_serial')
            if serial:
                devices = [d for d in devices if (d.getDeviceId() if self.major == 3 else d.getMxId()) == serial]
            if len(devices) != 1:
                raise RuntimeError(f'Expected exactly one selected OAK device, found {len(devices)}')
            w, h = int(config.get('width', 1920)), int(config.get('height', 1080))
            fps = float(config.get('fps', 15))
            socket = dai.CameraBoardSocket.CAM_A
            if self.major == 2:
                if (w, h) != (1920, 1080):
                    raise ValueError('v2 uses uncropped 1920x1080 video to preserve verified intrinsics mapping')
                self.pipeline = dai.Pipeline()
                cam = self.pipeline.create(dai.node.ColorCamera)
                cam.setBoardSocket(socket)
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                cam.setVideoSize(w, h)
                cam.setFps(fps)
                xout = self.pipeline.create(dai.node.XLinkOut)
                xout.setStreamName('rgb')
                cam.video.link(xout.input)
                self.device = dai.Device(self.pipeline, devices[0])
                self.queue = self.device.getOutputQueue('rgb', maxSize=1, blocking=False)
            elif self.major == 3:
                self.device = dai.Device(devices[0])
                self.pipeline = dai.Pipeline(self.device)
                # Default image: the 16:9 4K sensor mode the factory calibration was made
                # in, scaled uniformly to the output. Auto-detection would pick the full
                # 4:3 sensor and STRETCH it, so the factory intrinsics would not match.
                sensor = tuple(int(v) for v in config.get('sensor_resolution', (3840, 2160)))
                if sensor[0]*h != sensor[1]*w:
                    raise ValueError(f'Output {w}x{h} must keep the sensor {sensor[0]}x{sensor[1]} aspect ratio')
                cam = self.pipeline.create(dai.node.Camera).build(socket, sensorResolution=sensor)
                output = cam.requestOutput((w, h), type=dai.ImgFrame.Type.BGR888p,
                    resizeMode=dai.ImgResizeMode.STRETCH, fps=fps, enableUndistortion=False)
                self.queue = output.createOutputQueue(maxSize=1, blocking=False)
                self.pipeline.start()
                self.started = True
            else:
                raise RuntimeError(f'Unsupported DepthAI major version {self.major}')
            info = self.device.getDeviceInfo()
            self.serial = info.getDeviceId() if self.major == 3 else info.getMxId()
            calib = self.device.readCalibration()
            eeprom = calib.getEepromData()
            model = str(eeprom.cameraData[socket].cameraType).lower()
            self.board_name = eeprom.boardName
            self.product_name = eeprom.productName
            if 'perspective' not in model:
                raise ValueError(f'Unsupported OAK distortion model: {model}')
            _, calibrated_w, calibrated_h = calib.getDefaultIntrinsics(socket)
            if calibrated_w*h != calibrated_h*w:
                raise ValueError(f'Factory calibration {calibrated_w}x{calibrated_h} has a different aspect '
                                 f'ratio from the {w}x{h} output; factory intrinsics would not apply')
            self.K = np.asarray(calib.getCameraIntrinsics(socket, w, h, keepAspectRatio=self.major == 2), float)
            self.distortion = np.asarray(calib.getDistortionCoefficients(socket), float)
            self.size = (w, h)
        except Exception:
            self.close()
            raise

    def read(self, timeout_s=3.):
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline:
            if self.device.isClosed():
                raise ConnectionError('OAK disconnected; start a new session after reconnecting')
            packet = self.queue.tryGet()
            if packet is None:
                time.sleep(.005)
                continue
            stamp = packet.getTimestamp().total_seconds()
            sequence = packet.getSequenceNum()
            if self.previous is not None and (stamp <= self.previous[0] or sequence <= self.previous[1]):
                raise RuntimeError('OAK timestamp/sequence did not advance')
            self.previous = stamp, sequence
            image = packet.getCvFrame()
            if (image.shape[1], image.shape[0]) != self.size:
                raise ValueError('OAK output size differs from calibrated profile')
            # Preserve dai::Clock stamp. It is NOT Python monotonic without a verified mapping.
            return Frame(image, stamp, 'depthai_host_synced_clock', time.monotonic(),
                         self.config['optical_frame'], self.serial, self.K, self.distortion, sequence).validate()
        raise TimeoutError('OAK stream timed out')

    def close(self):
        if self.pipeline is not None and self.major == 3 and self.started:
            self.pipeline.stop()
        if self.device is not None:
            self.device.close()
