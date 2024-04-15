import ctypes
import logging
import json
import struct
import numpy as np

try:
    import tensorrt as trt
    from cuda import cuda

    TRT_SUPPORT = True
except ModuleNotFoundError:
    TRT_SUPPORT = False

from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum

logger = logging.getLogger(__name__)

DETECTOR_KEY = "tensorrt_ssd"

if TRT_SUPPORT:

    class TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            logger.log(self.getSeverity(severity), msg)

        def getSeverity(self, sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


class TensorRTDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: int = Field(default=0, title="GPU Device Index")


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem, nbytes, size):
        self.host = host_mem
        err, self.host_dev = cuda.cuMemHostGetDevicePointer(self.host, 0)
        self.device = device_mem
        self.nbytes = nbytes
        self.size = size

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        cuda.cuMemFreeHost(self.host)
        cuda.cuMemFree(self.device)


class TensorRtDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def _load_engine(self, model_path):
        try:
            trt.init_libnvinfer_plugins(self.trt_logger, "")
        except OSError as e:
            logger.error(
                "ERROR: failed to load libraries. %s",
                e,
            )

        logger.info(f'Loading model: {model_path}')
        with open(model_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_input_shape(self):
        """Get input shape of the TensorRT ssd engine."""
        binding = self.engine[0]
        assert self.engine.binding_is_input(binding)
        binding_dims = self.engine.get_binding_shape(binding)
        return (binding_dims[1:], trt.nptype(self.engine.get_binding_dtype(binding)),)

    def _allocate_buffers(self):
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        output_idx = 0
        for binding in self.engine:
            binding_dims = self.engine.get_binding_shape(binding)
            size = trt.volume(binding_dims)
            if size < 0:
                size = -size * 20  # size is dynamic, limit to 20x
            nbytes = size * self.engine.get_binding_dtype(binding).itemsize
            # Allocate host and device buffers
            err, host_mem = cuda.cuMemHostAlloc(
                nbytes, Flags=cuda.CU_MEMHOSTALLOC_DEVICEMAP
            )
            assert err is cuda.CUresult.CUDA_SUCCESS, f"cuMemAllocHost returned {err}"
            logger.debug(
                f"Allocated Tensor Binding {binding} Memory {nbytes} Bytes ({size} * {self.engine.get_binding_dtype(binding)})"
            )
            err, device_mem = cuda.cuMemAlloc(nbytes)
            assert err is cuda.CUresult.CUDA_SUCCESS, f"cuMemAlloc returned {err}"
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                logger.debug(f"Input has Shape {binding_dims}")
                inputs.append(HostDeviceMem(host_mem, device_mem, nbytes, size))
            else:
                # each grid has 3 anchors, each anchor generates a detection
                # output of 7 float32 values
                # assert size % 7 == 0, f"output size was {size}"
                logger.debug(f"Output has Shape {binding_dims}")
                outputs.append(HostDeviceMem(host_mem, device_mem, nbytes, size))
                output_idx += 1
        assert len(inputs) == 1, f"inputs len was {len(inputs)}"
        # assert len(outputs) == 1, f"output len was {len(outputs)}"
        return inputs, outputs, bindings

    def _do_inference(self):
        """do_inference (for TensorRT 8.0+)
        This function is generalized for multiple inputs/outputs for full
        dimension networks.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        # Push CUDA Context
        cuda.cuCtxPushCurrent(self.cu_ctx)

        # Transfer input data to the GPU.
        [
            cuda.cuMemcpyHtoDAsync(inp.device, inp.host, inp.nbytes, self.stream)
            for inp in self.inputs
        ]

        # Run inference.
        if not self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream
        ):
            logger.warn("Execute returned false")

        # Transfer predictions back from the GPU.
        [
            cuda.cuMemcpyDtoHAsync(out.host, out.device, out.nbytes, self.stream)
            for out in self.outputs
        ]

        # Synchronize the stream
        cuda.cuStreamSynchronize(self.stream)

        # Pop CUDA Context
        cuda.cuCtxPopCurrent()

        # Return only the host outputs.
        return [
            np.array(
                (ctypes.c_float * out.size).from_address(out.host), dtype=np.float32
            )
            for out in self.outputs
        ]

    def __init__(self, detector_config: TensorRTDetectorConfig):
        assert (
            TRT_SUPPORT
        ), f"TensorRT libraries not found, {DETECTOR_KEY} detector not present"

        (cuda_err,) = cuda.cuInit(0)
        assert (
            cuda_err == cuda.CUresult.CUDA_SUCCESS
        ), f"Failed to initialize cuda {cuda_err}"
        err, dev_count = cuda.cuDeviceGetCount()
        logger.debug(f"Num Available Devices: {dev_count}")
        assert (
            detector_config.device < dev_count
        ), f"Invalid TensorRT Device Config. Device {detector_config.device} Invalid."
        err, self.cu_ctx = cuda.cuCtxCreate(
            cuda.CUctx_flags.CU_CTX_MAP_HOST, detector_config.device
        )

        self.conf_th = 0.4  ##TODO: model config parameter
        err, self.stream = cuda.cuStreamCreate(0)
        self.trt_logger = TrtLogger()
        self.model_metadata = {}
        self.engine = self._load_engine(detector_config.model.path)
        self.input_shape = self._get_input_shape()
        self.model_type = detector_config.model.model_type
        
        assert self.model_type == ModelTypeEnum.ssd
        
        try:
            self.context = self.engine.create_execution_context()
            (
                self.inputs,
                self.outputs,
                self.bindings,
            ) = self._allocate_buffers()
        except Exception as e:
            logger.error(e)
            raise RuntimeError("fail to allocate CUDA resources") from e

        logger.debug("TensorRT loaded. Input shape is %s", self.input_shape)
        logger.debug("TensorRT version is %s", trt.__version__[0])

    def __del__(self):
        """Free CUDA memories."""
        if self.outputs is not None:
            del self.outputs
        if self.inputs is not None:
            del self.inputs
        if self.stream is not None:
            cuda.cuStreamDestroy(self.stream)
            del self.stream
        del self.engine
        del self.context
        del self.trt_logger
        cuda.cuCtxDestroy(self.cu_ctx)

    def detect_raw(self, tensor_input):
        if self.input_shape[-1] != trt.int8:
            tensor_input = tensor_input.astype(self.input_shape[-1])

        self.inputs[0].host = np.ascontiguousarray(
            tensor_input.astype(self.input_shape[-1])
        )

        trt_outputs = self._do_inference()
        count = int(trt_outputs[0][0])
        class_ids = [int(x) for x in trt_outputs[2]][:count]
        scores = trt_outputs[1][:count]
        # note: scalar scaling assumes square input for correct box
        boxes = (np.reshape(trt_outputs[3][:4*count], (count, 4)) * self.input_shape[0][0]).astype(int)

        detections = np.zeros((20, 6), np.float32)

        for i in range(count):
            if scores[i] < 0.4 or i == 20:
                break
            detections[i] = [
                class_ids[i],
                float(scores[i]),
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
            ]

        return detections
