#-*- coding: utf-8 -*-

import tensorflow as tf
from Data import stroke
from absl import app
import callbacks
from os.path import join
import numpy as np
import nibabel as nib

import flags
FLAGS = flags.FLAGS

########################################################################################################################
def main(*argv, **kwargs):
    a = tf.ones([1,2,2])
    print(a)

    return

if __name__ == '__main__':
    app.run(main)