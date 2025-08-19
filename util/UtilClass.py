import math
import os
import tempfile
import cv2
import numpy as np
import pycolmap
from pathlib import Path

import torch
from tqdm import tqdm
import h5py
from hloc import pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.localize_sfm import QueryLocalizer
from hloc.extractors.superpoint import SuperPoint as features_Model

from hloc.extractors.netvlad import NetVLAD as retrieval_Model

from hloc import extract_features, match_features
from hloc.pairs_from_retrieval import get_descriptors
from hloc.utils.io import list_h5_names
from hloc.utils.read_write_model import detect_model_format, read_images_binary, read_images_text
from util.utils import resize_features_for_match

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
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    # Global descriptors
    'netvlad': {
        'output': 'global-feats-netvlad',
        'model': {'name': 'netvlad'},
        'preprocessing': {'resize_max': 1024},
    }
}
def bron_kerbosch(R, P, X, graph):
    if len(P) == 0 and len(X) == 0:
        yield R
    while len(P) > 0:
        v = P[0]
        new_R = R + [v]
        new_P = [n for n in P if graph[v][n]]
        new_X = [n for n in X if graph[v][n]]
        yield from bron_kerbosch(new_R, new_P, new_X, graph)
        P.remove(v)
        X.append(v)

def find_maximal_cliques(graph):
    n = len(graph)
    P = list(range(n))
    R = []
    X = []
    return list(bron_kerbosch(R, P, X, graph))

def find_largest_clique(cliques):
    max_clique = []
    for clique in cliques:
        if len(clique) > len(max_clique):
            max_clique = clique
    return max_clique

def filter_cliques(cliques):
    unique_cliques = []
    used_nodes = set()
    for clique in sorted(cliques, key=len, reverse=True):
        if not any(node in used_nodes for node in clique):
            unique_cliques.append(clique)
            used_nodes.update(clique)
    return unique_cliques

class UtilClass(object):
    retrieval_conf = None
    feature_conf = None
    matcher_conf = None
    references = None
    model = None

    def __init__(self):
        pass

    def init(self, args):
        self.model = None
        self.xyz_scale = float(args.xyz_scale)
        self.images_loc = {}
        self.images = Path(args.input_path)
        self.outputs = Path(args.output_path) # 输出路径
        self.navigation_match_num = args.navigation_match_num # 导航匹配数
        self.model_names = [] # 模型名称（楼层）
        self.num_retrieval = args.num_retrieval # 检索数
        self.distance = args.distance # 距离限制
        self.use_rates = True if args.use_rates == 1 else False # 
        self.loc_pairs = self.outputs / 'pairs-loc.txt'
        self.retrieval_conf = extract_features.confs['netvlad']
        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.matcher_conf = match_features.confs['superglue']
        self.features_retrieval = self.outputs / 'features_retrieval.h5'
        self.features = self.outputs / 'features.h5'
        self.matches = self.outputs / 'matcher.h5'
        # Load the model-该场景下有几个场景，一般是指有几层楼
        for file_in_dir in os.listdir(self.outputs / 'sfm'):
            if os.path.isdir(os.path.join(self.outputs / 'sfm', file_in_dir)):
                self.model_names.append(file_in_dir)
        self.sfm_dir = [self.outputs / 'sfm' / model_name for model_name in self.model_names]

        # 加载colmap模型
        self.model = [pycolmap.Reconstruction(sfm_dir) for sfm_dir in self.sfm_dir]
        conf = {
            'estimation': {'ransac': {'max_error': 20}},
            'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
        }
        self.localizer = [QueryLocalizer(model, conf) for model in self.model]

        print("load model")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # retrieval_Model = dynamic_load(extractors, self.retrieval_conf['model']['name'])
        self.retrieval_model = retrieval_Model(self.retrieval_conf['model']).eval().to(device)

        # features_Model = dynamic_load(extractors, self.feature_conf['model']['name'])
        self.features_model = features_Model(self.feature_conf['model']).eval().to(device)

         # 读取检索场景的特征
        info = {"db_descriptors": [self.outputs / 'features_retrieval.h5']}
        info["name2db"] = {n: i for i, p in enumerate(info["db_descriptors"]) for n in list_h5_names(p)}
        info["db_names_h5"] = list(info["name2db"].keys())

        print("\n\n\n-----load images_from_model-----")
        info["db_names"] = []
        for sfm_dir in self.sfm_dir:
            if detect_model_format(sfm_dir, ".bin"):
                images = read_images_binary(sfm_dir / 'images.bin')
            elif detect_model_format(sfm_dir, ".txt"):
                images = read_images_text(sfm_dir / 'images.txt')
            # images = read_images_binary(sfm_dir / 'images.bin')
            # info["db_names"] = [i.name for i in images.values()]
            info["db_names"].extend([i.name for i in images.values()])
        info["db_desc"] = get_descriptors(info["db_names"], info["db_descriptors"], info["name2db"])
        info["db_names"] = info["db_names_h5"]
        info["db_desc"] = get_descriptors(info["db_names_h5"], info["db_descriptors"], info["name2db"])

        self.info = info
        sat_img = []
        for image_name in info["db_names"]:
            if "180" in image_name:
                sat_img.append(image_name)
        pairs_from_retrieval_info = {"db_descriptors": self.info["db_descriptors"],
                                        "name2db": {n: 0 for n in sat_img}}
        pairs_from_retrieval_info["db_names"] = list(pairs_from_retrieval_info["name2db"].keys())
        pairs_from_retrieval_info["db_desc"] = get_descriptors(pairs_from_retrieval_info["db_names"],
                                                                pairs_from_retrieval_info["db_descriptors"],
                                                                pairs_from_retrieval_info["name2db"])
        self.info = pairs_from_retrieval_info
        # 读取场景中每幅影像的特征点
        max_kpt_num = -1
        print("\n\n\n-----load features-----")
        self.images_features = {}
        with h5py.File(self.features, 'r') as f:
            mapping_groups = f["mapping"]
            for key in mapping_groups.keys():
                data = {}
                image = mapping_groups[key]
                for k, v in image.items():
                    data[k + '1'] = torch.from_numpy(v.__array__()).float().to(device)
                    # data[k + '1'] = torch.from_numpy(v.__array__()).float()
                # some matchers might expect an image but only use its size
                data['image1'] = torch.empty((1,) + tuple(image['image_size'])[::-1])
                self.images_features[f'mapping/{key}'] = data
        f.close()
        self.names2ref = {n: i for i, p in enumerate([self.features])
                          for n in list_h5_names(p)}
        self.max_kpt_num = max_kpt_num
        ### 获取场景中每幅影像的位置
        # 可以用于后续缓冲区分析
        max_xyz = [-9999, -9999, -9999]
        min_xyz = [9999, 9999, 9999]
        print("\n\n\n-----load image loc-----")
        with open(self.outputs / 'sfm' / "images_from_model.txt", "w") as f_w:
            for model, model_names in zip(self.model, self.model_names):
                for image in model.images.items():
                    # tvec = image[1].tvec
                    # qvec = image[1].qvec
                    # R = image[1].cam_from_world.rotation.matrix()
                    # quaternion = Rotation.from_matrix(R).as_quat()
                    # C = -np.dot(R.T, tvec)
                    pose = pycolmap.Image(tvec=image[1].tvec, qvec=image[1].qvec)
                    C = pose.projection_center()
                    # f_w.write(f'{image[1].name} {pose.projection_center()[0]} {pose.projection_center()[1]} '
                    #             f'{pose.projection_center()[2]} {model_names}\n')
                    f_w.write(f'{image[1].name} {C[0]} {C[1]} {C[2]} {model_names}\n')
        f_w.close()
        with open(self.outputs / 'sfm' / "images_from_model.txt") as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                images_info = line.split(" ")
                if images_info[0] in info["db_names"]:
                    self.images_loc[images_info[0]] = {
                        "xyz": [float(images_info[1]), float(images_info[2]), float(images_info[3])],
                        "model": images_info[4].replace("\n", "")
                    }
                    max_xyz = [max(data_[0], data_[1]) for data_ in zip(max_xyz, self.images_loc[images_info[0]]["xyz"])]
                    min_xyz = [min(data_[0], data_[1]) for data_ in zip(min_xyz, self.images_loc[images_info[0]]["xyz"])]
        fr.close()

        # 加载不同图像之间的重复率
        if self.use_rates:
            print("\n\n\n-----load images rates-----")
            self.rates = {}
            self.image_names = {}
            for model_path, model_name in zip(self.sfm_dir,self.model_names):
                self.image_names[model_name] = []
                rates = []
                with open(model_path / 'rates.txt', 'r') as f_r:
                    line = f_r.readline()
                    while line != "":
                        line_arr = line.replace("\n", "").split(":")
                        self.image_names[model_name].append(line_arr[0])
                        rates.append([float(data_) for data_ in line_arr[1].split(" ")])
                        line = f_r.readline()
                    self.rates[model_name] = np.array(rates)
    
    def find_img_in_which_db(self, pairs, loc_info, min_dis=10):
        pair_in_model = {}
        model_index = -1
        # 便利所有的匹配对，将匹配到的图片按照模型（楼层）进行分类
        for pair in pairs:
            if pair[1] not in pair_in_model.keys():
                pair_in_model[pair[1]] = [pair[0]]
            else:
                pair_in_model[pair[1]].append(pair[0])
        # 对匹配到的图片按照楼层进行筛选
        if loc_info["label"] is not None:
            model_names = [loc_info["label"]]
            pairs_new = pair_in_model[loc_info["label"]]
            model_index = self.model_names.index(loc_info["label"])
        else:
            model_names = list(pair_in_model.keys())
        loc_info["xyz"] = None
        for model_name in model_names:
            if loc_info["xyz"] is not None:
                pairs = []
                for pair in pairs_new:
                    dis = math.sqrt(sum(map(lambda x: x*x, [i-j for i, j in zip(self.images_loc[pair]['xyz'], loc_info['xyz'][-1])])))
                    if dis < min_dis:
                        pairs.append(pair)
            pairs_new = pairs
            most_pair = pairs_new

            if self.use_rates:
                pair_index = np.array([self.image_names[model_name].index(image_name[0]) for image_name in pairs_new])
                rates = self.rates[model_name][pair_index, :][:, pair_index]
                graph = rates > 0.55
                # graph = (rates > 0.5) & (ratess < 0.65)
                np.fill_diagonal(graph, 0)

                cliques = find_maximal_cliques(graph)
                unique_cliques = filter_cliques(cliques)
                largest_clique = find_largest_clique(unique_cliques)

                print("All maximal cliques:", cliques)
                print("Largest clique:", largest_clique)
                print("unique_cliques:", unique_cliques)
                print("Largest clique images:", [pairs_new[i] for i in largest_clique])
                result_pairs = []
                for unique_clique in unique_cliques:
                    if len(unique_clique) > 1:
                        result_pairs.append([pairs_new[i] for i in unique_clique])
            else:
                result_pairs = [pairs_new]
            # pairs_new =  [pairs_new[i] for i in largest_clique]
        
        return [None, None] if model_index == -1 else [self.model[model_index], self.localizer[model_index]], \
            result_pairs, self.model_names[model_index], most_pair
    
    def find_loc_by_image(self, image, query, limit_info):
        image = [{"image": image, "name": query}]
        query_features = self.outputs / 'query_features.h5'
        _, retrieval_descriptors = extract_features.main_my(self.retrieval_conf, self.images, self.retrieval_model, image,
                                                    image_list=[query],
                                                    feature_path=query_features, overwrite=True)
        _, match_descriptors = extract_features.main_my(self.feature_conf, self.images, self.features_model, image,
                                                image_list=[query],
                                                feature_path=self.features, overwrite=False)
        data0 = {}
        for key in match_descriptors.keys():
            data0[key + '0'] = torch.from_numpy(match_descriptors[key].__array__()).float().to('cuda')
        data0["image0"] = torch.empty((1,) + tuple(match_descriptors['image_size'])[::-1])

        num_retrieval = self.num_retrieval
        pairs_from_retrieval_info = self.info
        # 判断是否有位置限制
        if limit_info["xyz"] != None:
            last_loc = limit_info["xyz"][0]    
            sat_img = []
            for image_name in self.images_loc:
                image_loc = self.images_loc[image_name]["xyz"]
                dis = math.sqrt(
                    math.pow(image_loc[0] - last_loc[0], 2) + math.pow(image_loc[1] - last_loc[1], 2)
                    + math.pow(image_loc[2] - last_loc[2], 2))
                if dis < 10:
                    sat_img.append(image_name)
            pairs_from_retrieval_info = {"db_descriptors": self.info["db_descriptors"],
                                            "name2db": {n: 0 for n in sat_img}}
            pairs_from_retrieval_info["db_names"] = list(pairs_from_retrieval_info["name2db"].keys())
            pairs_from_retrieval_info["db_desc"] = get_descriptors(pairs_from_retrieval_info["db_names"],
                                                                    pairs_from_retrieval_info["db_descriptors"],
                                                                    pairs_from_retrieval_info["name2db"]) 
        pairs_new = pairs_from_retrieval.main_my(pred=retrieval_descriptors,
                                                 num_matched=num_retrieval,
                                                 info=pairs_from_retrieval_info, query_list=[query])
        pairs_new = [[image_name[1], self.images_loc[image_name[1]]['model']] for image_name in pairs_new if image_name[1] in self.images_loc.keys()]
        results = self.find_img_in_which_db(pairs_new, limit_info, self.distance)

        model = results[0][0]
        localizer  = results[0][1]
        model_name = results[2]
        most_pair = results[3]
        loc_list = {}
        most_pairis_in = {}
        match_num = 0
        for pairs_new in results[1]:
            pairs_new = pairs_new[:self.navigation_match_num]

            match_num += len(pairs_new)
            with open(self.loc_pairs, 'w') as f:
                f.write('\n'.join(' '.join([query, j[0]]) for j in pairs_new))
            f.close()
            self.names2ref[query] = 0
            _, matches_result = match_features.main_my(self.matcher_conf, self.loc_pairs, features=self.features,
                                                    matches=self.matches, data0=data0,
                                                    data1_list=self.images_features, overwrite=False,
                                                    names2ref=self.names2ref)
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=True)
            cv2.imwrite(temp_file.name, image[0]["image"])
            camera = pycolmap.infer_camera_from_image(temp_file.name)
            temp_file.close()
            ref_ids = [model.find_image_with_name(r[0]).image_id for r in pairs_new]

            ret, log, answer, kp_loc, kp_3d_loc = pose_from_cluster(localizer, query, camera, ref_ids, self.features,
                                                                    self.matches, match_descriptors["keypoints"], matches_result)
            print('inliers:', ret['num_inliers'])
            qvec = ret['qvec']
            pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
            print('pose:', pose.projection_center())
            loc_list[ret['num_inliers']] = [pose.projection_center(), qvec, model_name, 0, ret['num_inliers']]
            if ret['num_inliers'] > 50:
               most_pairis_in[ret['num_inliers']] = [pose.projection_center(), qvec, model_name, 0, ret['num_inliers']]
        # if len(most_pairis_in) > 0:
        #     dis = 9999
        #     for most_pair_key in most_pairis_in.keys():
        #         dis_now = math.sqrt(sum(map(lambda x: x*x, [i-j for i, j in zip(most_pairis_in[most_pair_key][0], last_loc)])))
        #         if dis_now < dis:
        #             dis = dis_now
        #             result = most_pairis_in[most_pair_key]
        #     return result
        keys = list(loc_list.keys())
        return loc_list[max(keys)], match_num