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
    from glob import glob
    FLAGS.inputs = glob('C:\dataset\\01_KUMC_data\\*\*\\*dwi_RPI_BFC*')
    for path in FLAGS.inputs:
    #     data = nib.load(path)
    #     header = data.header
    #     img_sz = header.get_data_shape()
        print(path)

    return

if __name__ == '__main__':
    app.run(main)