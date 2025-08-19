import base64
import cv2
from flask import json, jsonify
import numpy as np
import torch


def change_image(image, grayscale=False):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def processData(data, grayscale):
    image = change_image(data, grayscale)
    image = image.astype(np.float32)

    if grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))
    image = image / 255.
    return image


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    height, width = image_shape
    kpts = torch.tensor(np.ascontiguousarray(kpts).copy(), dtype=torch.float64)
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def process_superPoint_image(image):
    return (processData(image, True).reshape(1, 1, image.shape[0], image.shape[1]),
            processData(image, False).reshape(1, 3, image.shape[0], image.shape[1]))


def resize_features_for_match(data, shape):
    for i in range(len(data)):
        data[i][0] = normalize_keypoints(data[i][0], shape)
        data[i][1] = torch.tensor(np.reshape(data[i][1], (1,)+data[i][1].shape))
        data[i][2] = torch.tensor(np.reshape(data[i][2], (1,) + data[i][2].shape))
    return data


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = np.topk(scores, k, dim=0)
    return keypoints[indices], scores


def top_k_scores(scores, k: int):
    result = []
    for s in scores:
        if k >= len(s):
            result.append(s.argsort())
        else:
            result.append(s.argsort()[-1024:][::-1])
    return result


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints = torch.tensor(keypoints).float()
    descriptors = torch.tensor(descriptors)
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors.numpy()


def getSuperPointResultFromTensorrt(output):
    scores, descriptors = output
    _, h, w = scores.shape
    keypoints = [np.nonzero(s > 0.005) for s in scores]
    keypoints = [np.stack((keypoints[i][0], keypoints[i][1]), axis=1) for i in range(len(keypoints))]
    scores = [s[tuple(k.T)] for s, k in zip(scores, keypoints)]
    keypoints, scores = list(zip(*[
        remove_borders(k, s, 4, h, w)
        for k, s in zip(keypoints, scores)]))
    b = top_k_scores(scores, 1024)
    scores = [s[index] for s, index in zip(scores, b)]
    keypoints = [k[index, :] for k, index in zip(keypoints, b)]
    keypoints = [np.flip(k, [1]) for k in keypoints]
    descriptors = [sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, descriptors)]
    return keypoints, scores, descriptors



def process_superGlue_data(data0, data1_list, max_kpt_num):
    shape = []
    if data1_list[0][0].shape[1] < max_kpt_num:
        shape.append(data1_list[0][0].shape[1])
        data1_list[0][0] = torch.tensor(np.pad(data1_list[0][0], ((0, 0), (0, 1024 - data1_list[0][0].shape[1]), (0, 0)), mode='constant', constant_values=0))
        data1_list[0][1] = torch.tensor(np.pad(data1_list[0][1], ((0, 0), (0, 1024 - data1_list[0][1].shape[1])), mode='constant', constant_values=0))
        data1_list[0][2] = torch.tensor(np.pad(data1_list[0][2], ((0, 0), (0, 0), (0, 1024 - data1_list[0][2].shape[2])), mode='constant', constant_values=0))
    else:
        shape.append(max_kpt_num)
    data = data0 + data1_list[0]
    for i in range(1, len(data1_list)):
        if data1_list[i][0].shape[1] < max_kpt_num:
            shape.append(data1_list[i][0].shape[1])
            data1_list[i][0] = torch.tensor(np.pad(data1_list[i][0], ((0, 0), (0, 1024 - data1_list[i][0].shape[1]), (0, 0)), mode='constant', constant_values=0))
            data1_list[i][1] = torch.tensor(np.pad(data1_list[i][1], ((0, 0), (0, 1024 - data1_list[i][1].shape[1])), mode='constant', constant_values=0))
            data1_list[i][2] = torch.tensor(np.pad(data1_list[i][2], ((0, 0), (0, 0), (0, 1024 - data1_list[i][2].shape[2])), mode='constant', constant_values=0))
        else:
            shape.append(max_kpt_num)
        data[0] = torch.cat((data[0], data0[0]), dim=0)
        data[1] = torch.cat((data[1], data0[1]), dim=0)
        data[2] = torch.cat((data[2], data0[2]), dim=0)
        data[3] = torch.cat((data[3], data1_list[i][0]), dim=0)
        data[4] = torch.cat((data[4], data1_list[i][1]), dim=0)
        data[5] = torch.cat((data[5], data1_list[i][2]), dim=0)
    return data, shape


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def getSuperGlueResultFromTensorrt(scores, shape):
    scores = torch.tensor(scores[0])
    results = []
    for i in range(scores.shape[0]):
        scores_temp = scores[i, :, :shape[i]]
        scores_temp = scores_temp.reshape(1, -1, shape[i])
        max0, max1 = scores_temp[:, :-1, :-1].max(2), scores_temp[:, :-1, :-1].max(1)
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
        results.append({
            'matches0': indices0.numpy(),  # use -1 for invalid match
            # 'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0.numpy(),
            # 'matching_scores1': mscores1,
        })
    return results


def initLocByImageList_task(data, utilClass):
    images = data['images']
    image_ = []
    loc_info_list = []
    for image in images:
        image_binary = base64.b64decode(image)
        image_np = np.frombuffer(image_binary, dtype=np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_.append(image_cv)
        loc_info_list.append({"label": None, "xyz": None})
    loclist = utilClass.find_init_loc(image_, loc_info_list)
    results = []
    for loc in loclist:
        center, qvec, model_names, p = loc
        results.append({"x": center[0]*1000, "y": center[1]*1000, "z": center[2]*1000, "label": model_names, "qw": qvec[0],
                        "qx": qvec[1], "qy": qvec[2], "qz": qvec[3], "status": 200, "p": p})
    return jsonify(results)

def saveParams(data, fun_name):
    with open("./log.txt", "a") as f_a:
        f_a.write(fun_name +"__"+json.dumps(data)+"\n")
    f_a.close()
