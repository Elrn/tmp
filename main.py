import tensorflow as tf
import os, logging, re
from absl import app
import nibabel as nib
from Data import stroke
from os.path import join
import flags, utils, train
FLAGS = flags.FLAGS

########################################################################################################################
""" 전역변수 설정 """
########################################################################################################################
data = stroke

########################################################################################################################

def main(*argv, **kwargs):
    """  """
    """ init """
    utils.tf_init()
    paths = [FLAGS.ckpt_dir, FLAGS.plot_dir]
    [utils.mkdir(path) for path in paths]

    if FLAGS.train:
        train.main(argv, _dataset=data)

    else: # inference
        # saved_model_path = join(FLAGS.ckpt_dir, 'SavedModel.hdf5')
        saved_model_path = join(FLAGS.ckpt_dir, FLAGS.saved_model_name)
        model = tf.keras.models.load_model(saved_model_path) # custom_objects를 설정해줘야 한다.
        dataset, sizes = data.build_for_pred(FLAGS.inputs)

        if FLAGS.predict_step: # predict_on_batch 의 경우 tf.dataset을 받을 수 없음
            outputs = model.predict_step(dataset)

        else: # designed for batch processing of large numbers of inputs
            outputs = model.predict(
                dataset,
                batch_size=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False
            )
        """ output post-processing """
        data.post_processing(outputs, sizes)

########################################################################################################################
if __name__ == '__main__':
    app.run(main)

########################################################################################################################


