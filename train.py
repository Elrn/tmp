#-*- coding: utf-8 -*-

import tensorflow as tf
import utils, models, metrics, callbacks, losses
from Data import stroke
from tensorflow.keras.callbacks import *
import os, re, logging
from os.path import join
from absl import app

# import tensorflow.experimental.numpy as tnp
# tnp.experimental_enable_numpy_behavior()
########################################################################################################################
def main(*argv):
    if argv[0] == __file__:
        utils.tf_init()
    # init
    base_dir = os.path.dirname(os.path.realpath(__file__)) # getcwd()
    log_dir = join(base_dir, 'log')
    dirs = ['plt', 'checkpoint']
    paths = [join(log_dir, dir) for dir in dirs]
    [utils.mkdir(path) for path in paths]
    plt_dir, ckpt_dir = paths

    ### ckpt
    ckpt_file_name = 'EP_{epoch}, L_{loss:.3f}, P_{Precision:.3f}, R_{Recall:.3f}, J_{JSC:.3f}, vL_{val_loss:.3f}, vP_{val_Precision:.3f}, vR_{val_Recall:.3f}, vJ_{val_JSC:.3f}.hdf5'
    ckpt_file_path = join(ckpt_dir, ckpt_file_name)

    ### Get Data
    dataset, val_dataset = stroke.build(batch_size=5, validation_split=0.2) # [0]:train [1]:valid or None
    num_class, input_shape = stroke.num_class, stroke.input_shape

    ### Build model
    input = tf.keras.layers.Input(shape=input_shape)
    output = models.base(num_class)(input)
    # output = models.AGLN(num_class)(input)
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

    ### Train model
    history = model.fit(
        x = dataset,
        epochs=200,
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        callbacks=[
            ModelCheckpoint(ckpt_file_path, monitor='loss', save_best_only=True, save_weights_only=False, save_freq='epoch'),
            # EarlyStopping(monitor='loss', min_delta=0, patience=5),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, min_delta=0.0001, cooldown=0, min_lr=0),
            # callbacks.setLR(0.0001),
            # callbacks.monitor(plt_dir, dataset=test_dataset)
        ]
    )
    # date = datetime.datetime.today().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    # utils.save_history(history, utils.join_dir([base_dir, 'log', date]))

if __name__ == '__main__':
    app.run(main)