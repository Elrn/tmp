import os
from os.path import join, dirname
from absl import flags, logging

import utils

# logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS

set_log_verv = lambda debug:logging.set_verbosity(logging.DEBUG) if FLAGS.debug else logging.set_verbosity(logging.INFO)
base_dir = os.path.dirname(os.path.realpath(__file__))

"""
flags.register_validator('flag',
                         lambda value: value % 2 == 0,
                         message='some message when assert on')
flags.mark_flag_as_required('is_training')
"""

########################################################################################################################
""" model setting """
########################################################################################################################
flags.DEFINE_boolean('train', True, '모델 학습을 위한 모드')
flags.DEFINE_boolean('predict_step', True, '간단한 prediciton의 경우 사용, 대용량 prediction의 경우 false')


########################################################################################################################
""" Training settings """
########################################################################################################################
flags.DEFINE_integer("epochs", 10000, "")
flags.DEFINE_integer("bsz", 4, "Batch size")

########################################################################################################################
""" Dataset Setting """
########################################################################################################################


########################################################################################################################
""" Directory """
########################################################################################################################
log_dir = join(base_dir, 'log')
flags.DEFINE_string('_home_dir', os.path.expanduser('~'), 'home directory')
flags.DEFINE_string('_log_dir', join(os.path.expanduser('~'), 'log'), 'home directory')

flags.DEFINE_string('ckpt_dir', join(log_dir, 'checkpoint'), '체크포인트/모델 저장 경로')
flags.DEFINE_string('plot_dir', join(log_dir, 'plot'), 'plot 저장 경로')
flags.DEFINE_string('result_dir', join(log_dir, 'result'), '')

# flags.DEFINE_string('_log_dir', base_('log/'), '')

########################################################################################################################
""" Inference """
########################################################################################################################
flags.DEFINE_multi_string('inputs', None, 'list paths for prediction')