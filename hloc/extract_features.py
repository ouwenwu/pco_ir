import argparse
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
import h5py
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image
import time

from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor
from .utils.parsers import parse_image_lists
from .utils.io import read_image, list_h5_names, change_image

confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 1024,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    # Global descriptors
    'netvlad': {
        'output': 'global-feats-netvlad',
        'model': {'name': 'netvlad'},
        'preprocessing': {'resize_max': 1024},
    },
    'eigenplaces': {
        'output': 'global-feats-eigenplaces',
        'model': {'name': 'eigenplaces'},
        'preprocessing': {'resize_max': 1024},
    }
}


def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_' + interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        if paths is None:
            paths = []
            for g in conf.globs:
                paths += list(Path(root).glob('**/' + g))
            if len(paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            paths = sorted(list(set(paths)))
            self.names = [i.relative_to(root).as_posix() for i in paths]
            logger.info(f'Found {len(self.names)} images in root {root}.')
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p
                              for p in paths]
            else:
                raise ValueError(f'Unknown format for path argument {paths}.')

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(
                        f'Image {name} does not exists in root: {root}.')

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': name,
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)


class ImageDatasetMy(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, conf, images=None):
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.images = images

    def __getitem__(self, idx):
        image = self.images[idx]["image"]
        # image = read_image(self.root / name, self.conf.grayscale)
        image = change_image(image, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': self.images[idx]["name"],
            'image': image,
            'original_size': np.array(size),
        }
        end_time = time.time()
        return data

    def __len__(self):
        return len(self.images)

def process_netvlad_data(images):
    return_data = []
    for image_info in images:
        image = image_info['image']
        image = change_image(image, False)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        if max(size) > 1024:
            scale = 1024 / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.
        return_data.append({
            'name': [image_info['name']],
            'image': torch.tensor([image]),
            'original_size': torch.tensor([np.array(size)])
        })
    return return_data

def process_superpoint_data(images):
    return_data = []
    for image_info in images:
        image = image_info['image']
        image = change_image(image, True)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        if max(size) > 1024:
            scale = 1024 / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
        image = image[None]
        image = image / 255.
        return_data.append({
            'name': [image_info['name']],
            'image': torch.tensor([image]),
            'original_size': torch.tensor([np.array(size)])
        })
    return return_data

@torch.no_grad()
def main_my(conf: Dict,
            image_dir: Path,
            model,
            image,
            export_dir: Optional[Path] = None,
            as_half: bool = True,
            image_list: Optional[Union[Path, List[str]]] = None,
            feature_path: Optional[Path] = None,
            overwrite: bool = False):
    logger.info('Extracting local features with configuration:'
                f'\n{pprint.pformat(conf)}')
    # loader = ImageDatasetMy(conf['preprocessing'], image)
    # loader = torch.utils.data.DataLoader(loader, num_workers=1, batch_size=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if conf['model']['name'] == 'netvlad':
        process_data = process_netvlad_data(image)
    elif conf['model']['name'] == 'superpoint':
        process_data = process_superpoint_data(image)
    # data from loader 50ms
    for data in tqdm(process_data):
        # predict 57ms
        pred = model(map_tensor(data, lambda x: x.to(device)))
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:

            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, 'detection_noise', 1) * scales.mean()
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        # with h5py.File(str(feature_path), 'a', libver='latest') as fd:
        #     try:
        #         if name in fd:
        #             del fd[name]
        #         grp = fd.create_group(name)
        #         for k, v in pred.items():
        #             grp.create_dataset(k, data=v)
        #         if 'keypoints' in pred:
        #             grp['keypoints'].attrs['uncertainty'] = uncertainty
        #     except OSError as error:
        #         if 'No space left on device' in error.args[0]:
        #             logger.error(
        #                 'Out of disk space: storing features on disk can take '
        #                 'significant space, did you enable the as_half flag?')
        #             del grp, fd[name]
        #         raise error
        # del pred
    logger.info('Finished exporting features.')

    return feature_path, pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen',
                        choices=list(confs.keys()))
    parser.add_argument('--as_half', action='store_true')
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--feature_path', type=Path)
    args = parser.parse_args()
    main_my(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
