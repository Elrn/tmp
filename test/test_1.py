#-*- coding: utf-8 -*-

from absl import app
import flags
FLAGS = flags.FLAGS

########################################################################################################################
def main(*argv, **kwargs):
    print(kwargs['_dataset'])


if __name__ == '__main__':
    app.run(main)