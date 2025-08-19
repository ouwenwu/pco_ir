import argparse
import time
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
import collections.abc as collections

from hloc.matchers.superglue import SuperGlue as Model
from tqdm import tqdm
import h5py
import torch

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
from .utils.io import list_h5_names

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    }
}


def main_my(conf: Dict,
            pairs: Path, features: Union[Path, str],
            data0, data1_list,
            export_dir: Optional[Path] = None,
            matches: Optional[Path] = None,
            features_ref: Optional[Path] = None,
            overwrite: bool = False,
            names2ref=None):
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features + '.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    if isinstance(features_ref, collections.Iterable):
        features_ref = list(features_ref)
    else:
        features_ref = [features_ref]

    matches_result = match_from_paths_my(conf, pairs, matches, features_q, features_ref, data0, data1_list, overwrite, names2ref)

    return matches, matches_result


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths_my(conf: Dict,
                        pairs_path: Path,
                        match_path: Path,
                        feature_path_q: Path,
                        feature_paths_refs: Path,
                        data0, data1_list,
                        overwrite: bool = False,
                        name2ref=None):
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    match_result = {}
    for (name0, name1) in tqdm(pairs, smoothing=.1):
        data = data0.copy()

        data.update(data1_list[name1])
        data = {k: v[None] for k, v in data.items()}
        pred = model(data)
        pair = names_to_pair(name0, name1)
        match_result[pair] = {
            "matches": pred['matches0'][0].cpu().short().numpy(),
            "matching_scores": pred['matching_scores0'][0].cpu().half().numpy()
        }
    logger.info('Finished exporting matches.')
    torch.cuda.empty_cache()
    return match_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main_my(confs[args.conf], args.pairs, args.features, args.export_dir)
