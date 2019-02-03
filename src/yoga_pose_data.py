import argparse
import logging
import time
import glob
import ast
import os
import dill
import csv

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='/path/to/yoga/poses/directory/')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    yoga_dirs = os.listdir(args.folder)
    print(yoga_dirs)
    
    body_dict = {'Nose': 0 , 'Neck': 2, 'RShoulder': 4, 'RElbow':6, 'RWrist':8, 'LShoulder':10 , 'LElbow':12 , 'LWrist':14, 'RHip':16, 'RKnee':18, 'RAnkle':20, 'LHip':22, 'LKnee':24, 'LAnkle':26, 'REye':28, 'LEye':30, 'REar':32, 'LEar': 34 }

    with open("yoga_poses.csv", "w") as fl:
        writer = csv.writer(fl)
        for ydir in yoga_dirs:
            dd = args.folder + ydir + '/'
            files_grabbed = glob.glob(os.path.join(dd, '*'))
            all_humans = dict()
            for i, file in enumerate(files_grabbed):
                try:
                    # estimate human poses from a single image !
                    image = common.read_imgfile(file, None, None)
                    t = time.time()
                    humans = e.inference(image, scales=scales)
                    elapsed = time.time() - t

                    logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

                    body_arr = np.zeros(36)


                    if humans:
                        #for human in humans:
                        for parts in humans[0].body_parts.values():
                            bp = str(parts.get_part_name()).split('.')[1]
                            x = parts.x
                            y = parts.y
                            print(bp,x,y)
                            body_arr[body_dict[bp]], body_arr[body_dict[bp]+1] = x,y
                        print(body_arr)
                        row = body_arr.tolist()
                        row.append(ydir)
                        writer.writerow(row)
                except AttributeError:
                    print(str(file))
