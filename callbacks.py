from tensorflow.keras.callbacks import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import utils
import numpy as np
import re
import logging
from tensorflow.keras import backend
from keras.utils import io_utils

class monitor(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, dataset=None, fig_size_rate=3):
        super(monitor, self).__init__()
        self.save_dir = save_dir
        self.dataset = dataset
        self.fig_size_rate = fig_size_rate
    #
    def on_epoch_end(self, epoch, logs=None):
        # self.reconstuction_plot(epoch, logs)
        self.slices_plot(epoch, logs)

    def slices_plot(self, epoch, logs=None):
        epoch += 1
        cols, rows = 2, 4
        figure, axs = plt.subplots(cols, rows, figsize=(rows * 3, cols * 3))
        figure.suptitle(f'Epoch: "{epoch}"', fontsize=10)
        figure.tight_layout()
        for vol, seg in self.dataset.skip(15).take(1):
            pred = self.model(vol)
            vol, seg, pred = np.squeeze(vol), np.squeeze(tf.argmax(seg, -1)), np.squeeze(tf.argmax(pred, -1))

        for c in range(cols):
            for r in range(rows):
                axs[c][r].set_xticks([])
                axs[c][r].set_yticks([])
                if c == 0:
                    axs[c][r].imshow(vol[r], cmap='gray')
                    axs[c][r].imshow(seg[r], cmap='Greens', alpha=0.5)
                else:
                    axs[c][r].imshow(vol[r], cmap='gray')
                    axs[c][r].imshow(pred[r], cmap='Reds', alpha=0.5)

        save_path = os.path.join(self.save_dir, f'{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close('all')

    def patch_wise_plot(self, epoch):
        epoch += 1
        ncols = tf.data.experimental.cardinality(self.dataset).numpy() * 2
        nrows = self.dataset.element_spec[0].shape[0]

        figure, axs = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        figure.suptitle(f'{epoch}', fontsize=10)
        figure.tight_layout()
        slice = self.dataset.element_spec[0].shape[2] // 2
        # cmap = self.get_cmap()
        for i, (vol, seg) in enumerate(self.dataset):
            pred = self.model.predict_on_batch(vol)
            vol_, seg_, pred_ = vol, tf.argmax(seg, -1), tf.argmax(pred, -1)
            for r in range(nrows):
                vol, seg, pred = vol_[r], seg_[r], pred_[r]
                axs[i * 2][r].set_xticks([])
                axs[i * 2][r].set_yticks([])
                axs[(i * 2) + 1][r].set_xticks([])
                axs[(i * 2) + 1][r].set_yticks([])
                axs[i * 2][r].imshow(vol[:, :, slice], cmap='gray')
                axs[i * 2][r].imshow(seg[:, :, slice], cmap='Greens')
                axs[(i * 2) + 1][r].imshow(vol[:, :, slice], cmap='gray')
                axs[(i * 2) + 1][r].imshow(pred[:, :, slice], cmap='Reds', alpha=0.5)
                for spine in axs[(i * 2) + 1][r].spines.values():
                    spine.set_color('red')  # set Border color
        save_path = os.path.join(self.save_dir, f'patches_{epoch:05}.png')
        plt.savefig(save_path, dpi=150)
        plt.close('all')

    def reconstuction_plot(self, epoch):
        cols, rows = 2, 5
        figure, axs  = plt.subplots(cols, rows, figsize=(rows*3,cols*3))
        figure.tight_layout()
        vol, seg, pred = [], [], []
        for vol_, seg_ in self.dataset:
            pred_ = self.model.predict_on_batch(vol_)
            vol.append(vol_), seg.append(seg_), pred.append(pred_)
        vol, seg, pred = [tf.concat(arr, 0) for arr in (vol, seg, pred)]
        vol, seg, pred = [utils.reconstruct_pathces(arr, 1, [1,160,160,200,1], [1,40,40,40,1], [1,40,40,40,1])
                          for arr in (vol, seg, pred)]
        vol, seg, pred = np.squeeze(vol), np.squeeze(tf.argmax(seg, -1)), np.squeeze(tf.argmax(pred, -1))
        slice = int(vol.shape[2] * 0.7)
        s = 5
        for c in range(cols):
            for r in range(rows):
                axs[c][r].set_title(f'{slice + 4 * r}/{vol.shape[2]}', fontsize=7)
                axs[c][r].set_xticks([])
                axs[c][r].set_yticks([])
                if c == 0:
                    axs[c][r].imshow(vol[:, :, slice+s*r], cmap='gray')
                    axs[c][r].imshow(seg[:, :, slice+s*r], cmap='Greens', alpha=0.5)
                else:
                    axs[c][r].imshow(vol[:, :, slice+s*r], cmap='gray')
                    axs[c][r].imshow(pred[:, :, slice+s*r], cmap='Reds', alpha=0.5)

        save_path = os.path.join(self.save_dir, f'reconstuction_{epoch}.png')
        # figure.suptitle(f'{slice}/{vol.shape[2]} along Z axis', fontsize=5)
        plt.savefig(save_path, dpi=200)
        plt.close('all')

    def get_cmap(self):
        transparent = matplotlib.colors.colorConverter.to_rgba('white', alpha=0)
        white = matplotlib.colors.colorConverter.to_rgba('y', alpha=0.5)
        red = matplotlib.colors.colorConverter.to_rgba('r', alpha=0.7)
        return matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [transparent, white, red], 256)
#

class load_weights(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(load_weights, self).__init__()
        self.filepath = os.fspath(filepath) if isinstance(filepath, os.PathLike) else filepath

    def on_train_batch_begin(self, logs=None):
    # def on_predict_batch_begin(self, logs=None):
    # def on_predict_begin(self, logs=None):
        self.load_weights()

    def load_weights(self):
        filepath_to_load = (self._get_most_recently_modified_file_matching_pattern(self.filepath))
        if (filepath_to_load is not None and self.checkpoint_exists(filepath_to_load)):
            try:
                self.model.load_weights(filepath_to_load)
                print(f'[!] Saved Check point is restored from "{filepath_to_load}".')
            except (IOError, ValueError) as e:
                raise ValueError(f'Error loading file from {filepath_to_load}. Reason: {e}')

    @staticmethod
    def checkpoint_exists(filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        if filepath.endswith('.h5'):
            return tf.io.gfile.exists(filepath)
        tf_saved_model_exists = tf.io.gfile.exists(filepath)
        tf_weights_only_checkpoint_exists = tf.io.gfile.exists(
            filepath + '.index')
        return tf_saved_model_exists or tf_weights_only_checkpoint_exists

    @staticmethod
    def _get_most_recently_modified_file_matching_pattern(pattern):
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

        latest_tf_checkpoint = tf.train.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(
                base_name_regex, os.path.basename(latest_tf_checkpoint)):
            return latest_tf_checkpoint

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if tf.io.gfile.exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (file_path_with_largest_file_name is None or
                            file_path > file_path_with_largest_file_name):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found, reset
                        # the counter for the number of files with latest modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the most recent,
                        # increment the counter for the number of files with latest modified
                        # time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time, return
            # the file path with the largest file name.
            return file_path_with_largest_file_name


class setLR(Callback):
    def __init__(self, lr, verbose=0):
        super(setLR, self).__init__()
        self.lr = lr
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if not isinstance(self.lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             f'should be float. Got: {self.lr}')
        if isinstance(self.lr, tf.Tensor) and not self.lr.dtype.is_floating:
            raise ValueError(
                f'The dtype of `lr` Tensor should be float. Got: {self.lr.dtype}')
        backend.set_value(self.model.optimizer.lr, backend.get_value(self.lr))
        if self.verbose > 0:
            logging.info(
                f'\nEpoch {epoch + 1}: LearningRateScheduler setting learning '
                f'rate to {self.lr}.')
            logs = logs or {}
            logs['lr'] = backend.get_value(self.model.optimizer.lr)

# plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# early_stopping = EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
#     baseline=None, restore_best_weights=False
# )
# ckpt = ModelCheckpoint(
#     filepath, monitor='val_loss', verbose=0, save_best_only=False,
#     save_weights_only=False, mode='auto', save_freq='epoch',
# )

