import time
from pathlib import Path

import cv2
import tensorrt as trt
import numpy as np
import torch
from cuda import cudart
# cudart.cudaSetDevice(1)
from tqdm import tqdm

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1

def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def change_image(image, grayscale=False):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def getSuperGlueResultFromTensorrt(scores, kp_num_1, kp_num_2):
    scores = torch.tensor(scores[0])
    scores = scores[:, :kp_num_1, :kp_num_2]
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > 0.2)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    return {
        'matches0': indices0,  # use -1 for invalid match
        'matches1': indices1,  # use -1 for invalid match
        'matching_scores0': mscores0,
        'matching_scores1': mscores1,
    }


def superPointProcessData(data, grayscale):
    image = change_image(data, grayscale)
    image = image.astype(np.float32)

    if grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.
    return torch.tensor(image)


def superGlueProcessData(data, img_1_shape, img_2_shape):
    data[0] = torch.nn.functional.pad(data[0], (0, 0, 0, 1024-data[0].shape[0]), value=0)
    data[3] = torch.nn.functional.pad(data[3], (0, 0, 0, 1024-data[3].shape[0]), value=0)
    data[0] = normalize_keypoints(data[0], img_1_shape)
    data[1] = torch.reshape(data[1], (1,) + data[1].shape)
    data[1] = torch.nn.functional.pad(data[1], (0, 1024-data[1].shape[1]), value=0)
    data[2] = torch.reshape(data[2], (1,) + data[2].shape)
    data[2] = torch.nn.functional.pad(data[2], (0, 1024-data[2].shape[2]), value=0)
    data[3] = normalize_keypoints(data[3], img_2_shape)
    data[4] = torch.reshape(data[4], (1,) + data[4].shape)
    data[4] = torch.nn.functional.pad(data[4], (0, 1024-data[4].shape[1]), value=0)
    data[5] = torch.reshape(data[5], (1,) + data[5].shape)
    data[5] = torch.nn.functional.pad(data[5], (0, 1024-data[5].shape[2]), value=0)
    return data


class TensorrtModel:
    def __init__(self, conf):
        self.conf = conf
        self.nIO = None
        self.lTensorName = None
        self.nInput = None
        self.context = None
        self.engine = None
        self.bufferH = None
        self.bufferD = None
        self.shape = None
        self.loader_engine(conf["engine_path"], conf["model_input_shape"])

        # preload the model
        test_data = []
        for i in range(self.nInput):
            test_data.append(np.random.random(conf["model_input_shape"][i]))
        self.__call__(test_data)

    def loader_engine(self, engine_path, shape):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.nIO = self.engine.num_io_tensors
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(
            trt.TensorIOMode.INPUT)
        self.context = self.engine.create_execution_context()
        for i in range(self.nInput):
            self.context.set_input_shape(self.lTensorName[i], shape[i])
        self.shape = shape
        for i in range(self.nIO):
            print("[%2d]%s->" % (i, "Input" if i < self.nInput else "Output"),
                  self.engine.get_tensor_dtype(self.lTensorName[i]),
                  self.engine.get_tensor_shape(self.lTensorName[i]), self.context.get_tensor_shape(self.lTensorName[i]),
                  self.lTensorName[i])

    def mem_alloc(self, data):
        bufferH = []
        bufferD = []
        for i in range(self.nInput):
            self.context.set_input_shape(self.lTensorName[i], data[i].shape)
            bufferH.append(np.empty(self.context.get_tensor_shape(self.lTensorName[i]),
                                    dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))))
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
            np.copyto(bufferH[i], data[i])
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        for i in range(self.nInput, self.nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(self.lTensorName[i]),
                                    dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))))
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        for i in range(self.nIO):
            self.context.set_tensor_address(self.lTensorName[i], int(bufferD[i]))
        return bufferH, bufferD

    def __call__(self, data):
        bufferH, bufferD = self.mem_alloc(data)
        self.context.execute_async_v3(0)

        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        # keypoints, scores, descriptors = getResultFrom([self.bufferH[1], self.bufferH[2]])
        [cudart.cudaFree(b) for b in bufferD]
        del bufferD
        return bufferH[self.nInput:]


def draw_matches(kp1, kp2, matches0, matches1, img1, img2):
    matches = []
    for index, match0 in enumerate(matches0[0]):
        if match0 != -1:
            matches.append([index, int(match0)])
    matches = matches[5:10]
    img_combined = np.hstack((img1, img2))
    for match in matches:
        idx1, idx2 = match
        pt1 = (int(kp1[idx1][0]), int(kp1[idx1][1]))
        pt2 = (int(kp2[idx2][0])+img1.shape[1], int(kp2[idx2][1]))
        cv2.line(img_combined, pt1, pt2, (255, 0, 0), 1)
    cv2.imwrite("test2.jpg", img_combined)


if __name__ == "__main__":
    pass



