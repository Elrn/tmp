import numpy as np
import tensorflow as tf
import os, logging, re
from os.path import join
from absl import app
import nibabel as nib
from keras.utils import io_utils
from Data import stroke

import flags, utils, train, callbacks
FLAGS = flags.FLAGS

########################################################################################################################

def main(*argv):
    data = stroke
    utils.tf_init()

    if FLAGS.train:
        train.main(argv)


    else: # inference
        """ 모델 불러오기 """
        base_dir = os.path.dirname(os.path.realpath(__file__))  # getcwd()
        ckpt_dir = join(base_dir, 'log', 'tmp_model') if FLAGS.ckpt_dir==None else FLAGS.ckpt_dir
        # ckpt = callbacks.load_weights._get_most_recently_modified_file_matching_pattern(ckpt_dir)
        model = tf.keras.models.load_model(ckpt_dir)

        dataset, sizes = data.build_for_pred(FLAGS.inputs)


        if FLAGS.predict_step:
            output = model.predict_step(dataset)
            """
            predict_on_batch 의 경우 tf.dataset을 받을 수 없음
            """
        else: # designed for batch processing of large numbers of inputs
            output = model.predict(
                # x,
                batch_size=None,
                verbose='auto',
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False
            )
        #### output post_processing
        output = np.squeeze(np.argmax(output, -1))
        sep_inputs = []
        for size in sizes:
            sep_inputs.append(output[:size])
            output = output[size:]
        # output = np.transpose(output, [1, 2, 0])
        dir, basename = os.path.split(FLAGS.input)
        seg_file_name = 'seg_' + basename
        output_save_path = os.path.join(dir, seg_file_name)
        nib.save(output, output_save_path)

if __name__ == '__main__':
    app.run(main)


