#-*- coding: utf-8 -*-

import tensorflow as tf
from Data import stroke
from absl import app
import callbacks
from os.path import join
import numpy as np
import nibabel as nib
from tensorflow.keras.layers import *


import flags
FLAGS = flags.FLAGS

########################################################################################################################
def main(*argv, **kwargs):
    a = BatchNormalization()
    return

if __name__ == '__main__':
    app.run(main)