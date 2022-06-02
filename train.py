#-*- coding: utf-8 -*-

import tensorflow as tf
import utils, models, metrics, losses, callbacks
from Data import stroke
from tensorflow.python.keras.callbacks import *

import re
from os.path import join
from absl import app
import flags
FLAGS = flags.FLAGS

# import tensorflow.experimental.numpy as tnp
# tnp.experimental_enable_numpy_behavior()
########################################################################################################################
def main(*argv, **kwargs):
    if argv[0] == __file__:
        utils.tf_init()

    ### ckpt
    ckpt_file_path = join(FLAGS.ckpt_dir, FLAGS.ckpt_file_name)

    ### Get Data
    _dataset = stroke
    dataset, val_dataset = _dataset.build(batch_size=FLAGS.bsz, validation_split=0.2) # [0]:train [1]:valid or None
    num_class, input_shape = _dataset.num_class, _dataset.input_shape

    ### Build model
    input = tf.keras.layers.Input(shape=input_shape)
    output = models.base(num_class)(input)
    model = tf.keras.Model(input, output, name=None)

    ### Compile model
    metric_list = [
        metrics.Precision(num_class),
        metrics.Recall(num_class),
        metrics.F_Score(num_class),
        metrics.DSC(num_class),
        metrics.JSC(num_class),
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=losses.WCE(),
                  metrics=metric_list,
                  )

    ### load weights
    filepath_to_load = callbacks.load_weights._get_most_recently_modified_file_matching_pattern(ckpt_file_path)
    if (filepath_to_load is not None and callbacks.load_weights.checkpoint_exists(filepath_to_load)):
        initial_epoch = int(re.findall(r"EP_(\d+),", filepath_to_load)[0])
        try:
            model.load_weights(filepath_to_load)
            print(f'[Model|ckpt] Saved Check point is restored from "{filepath_to_load}".')
        except (IOError, ValueError) as e:
            raise ValueError(f'Error loading file from {filepath_to_load}. Reason: {e}')
    else:
        print(f'[Model|ckpt] Model is trained from scratch.')
        initial_epoch = 0

    _callbacks=[
        ModelCheckpoint(ckpt_file_path, monitor='loss', save_best_only=True, save_weights_only=False, save_freq='epoch'),
        # EarlyStopping(monitor='loss', min_delta=0, patience=5),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, min_delta=0.0001, cooldown=0, min_lr=0),
        # callbacks.setLR(0.0001),
    ]
    if FLAGS.plot:
        adc_dataset, dwi_dataset = _dataset.build_test(FLAGS.bsz)
        _callbacks.append(callbacks.monitor(FLAGS.plot_dir, dataset=adc_dataset))

    ### Train model
    history = model.fit(
        x=dataset,
        epochs=200,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        callbacks=_callbacks,
    )
    if FLAGS.save == True:
        save_path = join(FLAGS.ckpt_dir, FLAGS.saved_model)
        model.save(save_path)
    # date = datetime.datetime.today().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    # utils.save_history(history, utils.join_dir([base_dir, 'log', date]))

if __name__ == '__main__':
    app.run(main)