import tensorflow as tf
import utils, models, metrics, callbacks, losses
from Data import stroke
from tensorflow.keras.callbacks import *
import os, re, logging
from os.path import join
from absl import app
import flags
FLAGS = flags.FLAGS
import numpy as np


def main(arg):
    ckpt_dir = join(flags.base_dir, 'log', 'tmp_model')
    model = tf.keras.models.load_model(ckpt_dir)

    # ckpt = callbacks.load_weights._get_most_recently_modified_file_matching_pattern(ckpt_dir)
    # _dataset = stroke
    # ds = _dataset.build_for_pred(FLAGS.inputs)
    #
    # for d in ds:
    #     print(d.shape)

if __name__ == '__main__':
    app.run(main)