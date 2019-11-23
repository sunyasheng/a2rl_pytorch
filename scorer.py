import tensorflow as tf
import os
import numpy as np
from envs_zoo import vfn_network as vfn_net
import skimage.transform as transform

class Scorer(object):
    def __init__(self, args):
        self.ranking_loss = args.ranking_loss
        self.global_dtype_np = np.float32
        self.img_h, self.img_w = 227, 227
        
        embedding_dim = args.embedding_dim
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        net_data_pth = os.path.join(cur_dir, args.initial_parameters)
        
        net_data = np.load(net_data_pth, encoding='latin1').item()
        SPP = args.spp
        pooling = args.pooling
        global_dtype = tf.float32
        self.batch_size = 1
        self.image_placeholder = tf.placeholder(dtype=global_dtype, shape=[self.batch_size,self.img_h,self.img_w,3])
        var_dict = vfn_net.get_variable_dict(net_data)
        
        with tf.variable_scope("ranker") as scope:
            self.feature_vec = vfn_net.build_alexconvnet(self.image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
            self.score_func = vfn_net.score(self.feature_vec)

        self.saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session(config=tf.ConfigProto())
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, args.oracle_model_pth)

    def score_feature(self, img):
        img = img.astype(np.float32) / 255
        img_resize = transform.resize(img, (self.img_h, self.img_w)) - 0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        score, feature = self.sess.run([self.score_func, self.feature_vec], feed_dict={self.image_placeholder: img_resize})
        return score, feature
