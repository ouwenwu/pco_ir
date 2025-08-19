import argparse
import base64
import datetime
import time

import cv2
from flask import Flask, jsonify, request
import numpy as np

from util.utils import saveParams
from util.UtilClass import UtilClass


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="datasets", required=False,
                    help="Input path (the path where the image is stored)")
parser.add_argument('--output_path', type=str, default="outputs", required=False, help="Output path")
parser.add_argument('--port', type=int, default=5001, required=False, help="Host port")
parser.add_argument('--host', type=str, default="127.0.0.1", required=False, help="Host IP")
parser.add_argument('--num_retrieval', type=int, default=20, required=False, help="Retrieves the number of images")
parser.add_argument('--init_match_num', type=int, default=20, required=False,
                    help="The number of matches at initialization")
parser.add_argument('--navigation_match_num', type=int, default=10, required=False,
                    help="The number of matches during navigation")
parser.add_argument('--distance', type=int, default=10, required=False,
                    help="Distance restriction policy")
parser.add_argument('--xyz_scale', type=int, default=1, required=False,
                    help="xyz_scale")
parser.add_argument('--use_rates', type=int, default=1, required=False,
                    help="use_rates")
args = parser.parse_args()

utilClass = UtilClass()
utilClass.init(args)

app = Flask(__name__)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/getlocByImage', methods=['POST']) 
def getlocByImage():
    try:
        print('getlocByImage')
        data = request.get_json()
        
        # 把参数保存下来
        saveParams(data, "getlocByImage")

        image = data['image']
        # 是否有上一次的位置做参考
        if 'last_loc' in data.keys():
            last_loc = data['last_loc']
        else:
            last_loc = None
        
        # 是否有标签（楼层）对图片做限制
        label = data['floor'] if 'floor' in data.keys() else None

        # 用于定位的图片
        image_binary = base64.b64decode(image)
        image_np = np.frombuffer(image_binary, dtype=np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_ = image_cv

        # 保存当前的图片
        now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        query = "query/" + now_time + ".jpg"
        cv2.imwrite("./test/"+str(time.time())+".jpg", image_cv)

        # 将对图像的限制值（last_loc、label）组织在一起
        loc_info = {"label": label, "xyz": last_loc}
        [center, qvec, model_names, p, _], match_num = utilClass.find_loc_by_image(image_, query, loc_info)
        results = [{"x": center[0]*utilClass.xyz_scale, "y": center[1]*utilClass.xyz_scale, "z": center[2]*utilClass.xyz_scale, "label": model_names, "qw": qvec[0],
                    "qx": qvec[1], "qy": qvec[2], "qz": qvec[3], "status": 200, "p": p, "match_num": match_num}]
        return jsonify(results)
        

    except Exception as e:
        print(e)
        return str(e)


if __name__ == '__main__':
    app.run(port=args.port, host=args.host, debug=False)
