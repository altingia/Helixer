#! /usr/bin/env python3
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, CuDNNLSTM, Dense, Bidirectional
from HelixerModel import HelixerModel, get_col_accuracy_fn


class DanQModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('-u', '--units', type=int, default=4)
        self.parser.add_argument('-f', '--filter-depth', type=int, default=8)
        self.parser.add_argument('-ks', '--kernel-size', type=int, default=26)
        self.parse_args()

    def model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.filter_depth,
                         kernel_size=self.kernel_size,
                         input_shape=(self.shape_train[1], 4),
                         padding="same",
                         activation="relu"))

        model.add(Bidirectional(CuDNNLSTM(self.units, return_sequences=True)))

        model.add(Dense(3, activation='sigmoid'))
        return model

    def compile_model(self, model):
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      metrics=[
                          'accuracy',
                          get_col_accuracy_fn(0),
                          get_col_accuracy_fn(1),
                          get_col_accuracy_fn(2),
                      ])

    # generator should be the same as for the cnn
    def _gen_data(self, h5_file, shuffle, exclude_err_seqs=False, sample_intergenic=False):
        n_seq = h5_file['/data/X'].shape[0]
        if exclude_err_seqs:
            err_samples = np.array(h5_file['/data/err_samples'])
        if sample_intergenic and self.intergenic_chance < 1.0:
            fully_intergenic_samples = np.array(h5_file['/data/fully_intergenic_samples'])
            intergenic_rolls = np.random.random((n_seq,))  # a little bit too much, but simpler so
        X, y = [], []
        while True:
            seq_indexes = list(range(n_seq))
            if shuffle:
                random.shuffle(seq_indexes)
            for n, i in enumerate(seq_indexes):
                if exclude_err_seqs and err_samples[i]:
                    continue
                if (sample_intergenic and self.intergenic_chance < 1.0
                        and fully_intergenic_samples[i]
                        and intergenic_rolls[i] > self.intergenic_chance):
                    continue
                X.append(h5_file['/data/X'][i])
                y.append(h5_file['/data/y'][i])
                # apply intergenic sample weight value
                if n == len(seq_indexes) - 1 or len(X) == self.batch_size:
                    yield (
                        np.stack(X, axis=0),
                        np.stack(y, axis=0)
                    )
                    X, y = [], []


if __name__ == '__main__':
    model = DanQModel()
    model.run()
