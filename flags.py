import os
from os.path import join, dirname
from absl import flags, logging

import utils

# logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS

set_log_verv = lambda debug:logging.set_verbosity(logging.DEBUG) if FLAGS.debug else logging.set_verbosity(logging.INFO)

base_dir = os.path.dirname(os.path.realpath(__file__))

base_ = lambda dir:join(base_dir, dir)
home_ = lambda dir:join(FLAGS.home_dir, dir)

"""
flags.register_validator('flag',
                         lambda value: value % 2 == 0,
                         message='some message when assert on')
flags.mark_flag_as_required('is_training')
"""


########################################################################################################################
""" model setting """
########################################################################################################################
flags.DEFINE_string('model', '', 'Select model to Build')
flags.DEFINE_boolean('train', True, '모델 학습을 위한 모드')

#
flags.DEFINE_boolean("from_scratch", False, "")
flags.DEFINE_boolean('test', False, '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('predict_step', False, '')


########################################################################################################################
""" Training settings """
########################################################################################################################
flags.DEFINE_integer("epochs", 10000, "")
flags.DEFINE_integer("batch_size", 2, "")
flags.DEFINE_boolean("schedule_lr", False, "Use Schedule learning rate")

flags.DEFINE_integer('save_frequency', 2, 'save frequency during training', lower_bound=2)
flags.DEFINE_integer('valid_frequency', 2, 'validation frequency during training', lower_bound=2)

# for GAN Model
flags.DEFINE_float('lambd', 10.0, 'HyperPram for Penalty for critic update', lower_bound=0.1, upper_bound=10.0)
flags.DEFINE_float("l1_map_weight", 10.0, "used for generator losses function weight")

# flags.DEFINE_integer('decay_steps', 10, 'schedule learning rate frequency')
# flags.DEFINE_string('valid_file', 'VALID', 'Validation checkpoint 파일 이름')
# flags.DEFINE_integer('valid_count', 30, 'number of validation data')

########################################################################################################################
""" Dataset Setting """
########################################################################################################################
flags.DEFINE_list('input_shape', [140, 200, 160, 1], 'input_shape')
flags.DEFINE_list('image_shape', [140, 200, 160], 'shape of image')

# flags.DEFINE_integer('len_dataset', 399, '', lower_bound=0)

########################################################################################################################
""" Optimizer Setting """
########################################################################################################################
flags.DEFINE_float('lr', 5e-5, 'learning rate', lower_bound=0)
flags.DEFINE_float('beta1', 0.0, 'args for adam optimizer', lower_bound=0)
flags.DEFINE_float('beta2', 0.9, 'args for adam optimizer', lower_bound=0)

########################################################################################################################
""" Directory """
########################################################################################################################
flags.DEFINE_string('home_dir', os.path.expanduser('~'), 'home directory')

flags.DEFINE_string('ckpt_dir', base_('checkpoint'), '체크포인트/모델 저장 경로')
flags.DEFINE_string('plot_dir', base_('plot'), 'plot 저장 경로')
flags.DEFINE_string('result_dir', base_('result'), '')

# flags.DEFINE_string('_log_dir', base_('log/'), '')

########################################################################################################################
