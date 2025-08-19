import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
from . import logger
from .utils.parsers import parse_image_lists
from .utils.io import list_h5_names


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(f'Unknown type of image list: {names}.'
                             'Provide either a list or a path to a list file.')
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
    if name2idx is None:
        with h5py.File(str(path), 'r', libver='latest') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), 'r', libver='latest') as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(scores: torch.Tensor,
                            invalid: np.array,
                            num_select: int,
                            min_score: Optional[float] = None):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float('-inf'))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def main_my(num_matched, info, pred, query_list=None):
    logger.info('Extracting image pairs from a retrieval database.')
    if len(info['name2db']) < num_matched:
        pairs = [(query_list[0], j) for j in info['db_names']]
    else:
        # query_names_h5 = list_h5_names(descriptors)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # query_names = parse_names(query_prefix, query_list, query_names_h5)
        # query_desc = get_descriptors(query_names, descriptors)
        query_names = query_list
        query_desc = torch.from_numpy(np.stack([pred["global_descriptor"]], 0)).float()
        sim = torch.einsum('id,jd->ij', query_desc.to(device), info["db_desc"].to(device))
        # Avoid self-matching
        self = np.array(query_names)[:, None] == np.array(info["db_names"])[None]
        pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
        pairs = [(query_names[i], info["db_names"][j]) for i, j in pairs]

    logger.info(f'Found {len(pairs)} pairs.')
    # print('Extracting image pairs from a retrieval database.')
    # query_desc = torch.from_numpy(np.stack([pred["global_descriptor"]], 0)).float()
    # sim = np.einsum('id,jd->ij', query_desc, info["db_desc"])
    # if len(invalid) == 0:
    #     self = np.array([str("") for _ in range(pred["global_descriptor"].shape[0])])[:, None] == np.array(info["db_names"])[None]
    #     pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    # else:
    #     pairs = pairs_from_score_matrix(sim, np.array(invalid).reshape(1, len(invalid)), num_matched, min_score=0)
    # pairs = [info["db_names"][i[1]] for index, i in enumerate(pairs)]
    # print(f'Finish {len(pairs)} image pairs.')
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()
    main_my(**args.__dict__)
