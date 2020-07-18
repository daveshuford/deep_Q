'''
Doing some DQN - OOP
Tensorboard MOD - Big Deal when training
'''
from keras.callbacks import TensorBoard  # May or MayNot need this
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from collections import deque

import os

'''
Tensorboard wants to create a log-file for every .fit [ THIS IS PROBABLY WHAT HAPPENED TO MY IMAGE RUNS ] 
- since we are doing that for every-single node (256) we 
don't want this to happen - so below we override it
'''
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)


    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)