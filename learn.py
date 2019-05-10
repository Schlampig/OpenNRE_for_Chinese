# -*- coding: utf-8 -*-

import os
from prepare import *
from config import Config
import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
MODEL = {'pcnn_att': models.PCNN_ATT,
      'pcnn_one': models.PCNN_ONE,
      'pcnn_ave': models.PCNN_AVE,
      'cnn_att': models.CNN_ATT,
      'cnn_one': models.CNN_ONE,
      'cnn_ave': models.CNN_AVE}


def learn(model_name="pcnn_att", num_class=0, epoch=-1, is_training=True):
    if num_class == 0 or epoch <= 0:
        raise Exception("[ERROR] Wrong number of class / epoch. (num_class>0, epoch>0)")
    con = Config(batch_size=64, 
             max_epoch=epoch, 
             num_class=num_class, 
             max_length=120, 
             is_training=is_training)
    if is_training:
        con.load_train_data()
        con.load_test_data()
        con.set_train_model(MODEL[model_name])
        con.train()
    else:
        con.load_test_data()
        con.set_test_model(MODEL[model_name])
        con.set_epoch_range([10, 13, 15])
        con.test()
    return None


if __name__ == "__main__":
    learn(model_name="pcnn_att",
         epoch=20,
         num_class=51,
         is_training=False)
