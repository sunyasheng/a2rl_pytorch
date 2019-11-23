from .actions import command2action, generate_bbox, crop_input
import numpy as np
import argparse
import glob
import os
import skimage.io as io
import skimage.transform as transform
import torch

class Environment(object):
    def __init__(self, args, scorer):
        self.scorer = scorer
        self.image_paths = []
        with open(args.image_train_list, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                cur_path = line.strip()
                self.image_paths.append(cur_path)
        print("total {} pics in this env!!".format(len(self.image_paths)))
        # self.image_paths = glob.glob(os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), args.image_dir, '*.jpg'))
        # self.image_paths = glob.glob(os.path.join(args.image_dir, '*.jpg'))
        # print(self.image_paths)
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # self.snapshot_pth = args.oracle_model_pth
        # self.snapshot_pth = os.path.join(cur_dir, os.path.pardir, args.snapshot)
        # if not os.path.exists(self.snapshot_pth):
        #     os.makedirs(self.snapshot_pth)
        
        # import pdb; pdb.set_trace()
        # self.sess = self.scorer.init(self.snapshot_pth)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.origin_img_path = self.image_paths[np.random.choice(len(self.image_paths))]
        # print(self.origin_img_path)
        # import pdb; pdb.set_trace();
        self.origin_img = io.imread(self.origin_img_path)[:, :, :3]
        self.origin_score, self.origin_feature = self.scorer.score_feature(self.origin_img)
        self.cur_score = self.origin_score
        self.batch_size = 1
        self.ratios = np.repeat([[0, 0, 20, 20]], self.batch_size, axis=0)
        self.dones = np.repeat([0], self.batch_size, axis=0)
        self.steps = 0
        return np.concatenate([self.origin_feature]*2, axis=1)

    def step(self, act_np):
        self.steps += 1
        self.ratios, self.dones = command2action(act_np, self.ratios, self.dones)
        bbox = generate_bbox(np.expand_dims(self.origin_img, axis=0), self.ratios)
        # import pdb; pdb.set_trace();
        next_img = crop_input(np.expand_dims(self.origin_img, axis=0), bbox)
        score, ob_np = self.scorer.score_feature(next_img[0])
        self.croped_bbox = bbox[0]
        reward = 1 if score > self.cur_score else -1
        self.cur_score = score
        reward -= 0.001*self.steps
        reward -= 5 if (bbox[0][2] - bbox[0][0]) > 2*(bbox[0][3] - bbox[0][1]) \
                or (bbox[0][2] - bbox[0][0]) < 0.5*(bbox[0][3] - bbox[0][1]) else 0
        assert np.all(self.dones==1) == np.any(self.dones==1), "batch size is not 1"
        if np.all(self.dones==1):
            reward = 0
        return np.concatenate([self.origin_feature, ob_np], axis=1), reward, np.all(self.dones==1), []

    def cur_status(self):
        return self.origin_img, self.croped_bbox, self.cur_score - self.origin_score