#! /usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import tensorflow as tf

from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras.layers import (LSTM, CuDNNLSTM, Dense, Bidirectional, Dropout, Reshape, Activation,
                          Input)
from helixer.prediction.HelixerModel import HelixerModel, HelixerSequence


class LSTMSequence(HelixerSequence):
    def __init__(self, model, h5_file, mode, shuffle):
        super().__init__(model, h5_file, mode, shuffle)
        if self.class_weights is not None:
            assert not mode == 'test'  # only use class weights during training and validation
        if self.error_weights:
            assert not mode == 'test'

    def __getitem__(self, idx):
        X, y, sw, error_rates, gene_lengths, transitions, coverage_scores = self._get_batch_data(idx)
        pool_size = self.model.pool_size
        assert pool_size > 1, 'pooling size of <= 1 oh oh..'
        assert y.shape[1] % pool_size == 0, 'pooling size has to evenly divide seq len'

        X = X.reshape((
            X.shape[0],
            X.shape[1] // pool_size,
            -1
        ))
        # make labels 2d so we can use the standard softmax / loss functions
        y = y.reshape((
            y.shape[0],
            y.shape[1] // pool_size,
            pool_size,
            y.shape[-1],
        ))

        # mark any multi-base timestep as error if any base has an error
        sw = sw.reshape((sw.shape[0], -1, pool_size))
        sw = np.logical_not(np.any(sw == 0, axis=2)).astype(np.float32)

        # only change sample weights during training (not even validation) as we don't calculate
        # a validation loss at the moment
        if self.mode == 'train':
            if self.class_weights is not None:
                # class weights are only used during training and validation to keep the loss
                # comparable and are additive for the individual timestep predictions
                # giving even more weight to transition points
                # class weights without pooling not supported yet
                # cw = np.array([0.8, 1.4, 1.2, 1.2], dtype=np.float32)
                cls_arrays = [np.any((y[:, :, :, col] == 1), axis=2) for col in range(4)]
                cls_arrays = np.stack(cls_arrays, axis=2).astype(np.int8)
                # add class weights to applicable timesteps
                cw_arrays = np.multiply(cls_arrays, np.tile(self.class_weights, y.shape[:2] + (1,)))
                cw = np.sum(cw_arrays, axis=2)
                # multiply with previous sample weights
                sw = np.multiply(sw, cw)

            if self.gene_lengths:
                gene_lengths = gene_lengths.reshape((gene_lengths.shape[0], -1, pool_size))
                gene_lengths = np.max(gene_lengths, axis=-1)  # take the maximum per pool_size (block)
                # scale gene_length to a sample weight that is 1 for the average
                gene_idx = np.where(gene_lengths)
                ig_idx = np.where(gene_lengths == 0)
                gene_weights = gene_lengths.astype(np.float32)
                scaled_gene_lengths = self.gene_lengths_average / gene_lengths[gene_idx]
                # the exponent controls the steepness of the curve
                scaled_gene_lengths = np.power(scaled_gene_lengths, self.gene_lengths_exponent)
                scaled_gene_lengths = np.clip(scaled_gene_lengths, 0.1, self.gene_lengths_cutoff)
                gene_weights[gene_idx] = scaled_gene_lengths.astype(np.float32)
                # important to set all intergenic weight to 1
                gene_weights[ig_idx] = 1.0
                sw = np.multiply(gene_weights, sw)

            if self.transition_weights is not None:
                transitions = transitions.reshape((
                    transitions.shape[0],
                    transitions.shape[1] // pool_size,
                    pool_size,
                    transitions.shape[-1],
                ))
                # more reshaping and summing  up transition weights for multiplying with sample weights
                sw_t = self.compress_tw(transitions)
                sw = np.multiply(sw_t, sw)

            if self.coverage_weights:
                coverage_scores = coverage_scores.reshape((coverage_scores.shape[0], -1, pool_size))
                # maybe offset coverage scores [0,1] by small number (bc RNAseq has issues too), default = 0.0
                if self.coverage_offset > 0.:
                    coverage_scores = np.add(coverage_scores, self.coverage_offset)    
                coverage_scores = np.mean(coverage_scores, axis=2)
                sw = np.multiply(coverage_scores, sw)

            if self.error_weights:
                # finish by multiplying the sample_weights with the error rate
                # 1 - error_rate^(1/3) seems to have the shape we need for the weights
                # given the error rate
                error_weights = 1 - np.power(error_rates, 1/3)
                sw *= np.expand_dims(error_weights, axis=1)

        return X, y, sw

    def compress_tw(self, transitions):
        return self._squish_tw_to_sw(transitions, self.transition_weights, self.stretch_transition_weights)

    @staticmethod
    def _squish_tw_to_sw(transitions, tw, stretch):
        sw_t = [np.any((transitions[:, :, :, col] == 1), axis=2) for col in range(6)]
        sw_t = np.stack(sw_t, axis=2).astype(np.int8)
        sw_t = np.multiply(sw_t, tw)

        sw_t = np.sum(sw_t, axis=2)
        where_are_ones = np.where(sw_t == 0)
        sw_t[where_are_ones[0], where_are_ones[1]] = 1
        if stretch is not 0:
            sw_t = LSTMSequence._apply_stretch(sw_t, stretch)
        return sw_t

    @staticmethod    
    def _apply_stretch(reshaped_sw_t, stretch):
        """modifies sample weight shaped transitions so they are a peak instead of a single point"""
        reshaped_sw_t = np.array(reshaped_sw_t)  
        dilated_rf = np.ones(np.shape(reshaped_sw_t))  
        
        where = np.where(reshaped_sw_t > 1)
        i = np.array(where[0])  # i unchanged
        j = np.array(where[1])  # j +/- step
    
        # find dividers depending on the size of the dilated rf
        dividers = []
        for distance in range(1, stretch + 1):
            dividers.append(2**distance)
        
        for z in range(stretch, 0, -1):
            dilated_rf[i, np.maximum(np.subtract(j, z), 0)] = np.maximum(reshaped_sw_t[i, j]/dividers[z-1], 1)
            dilated_rf[i, np.minimum(np.add(j, z), len(dilated_rf[0])-1)] = np.maximum(reshaped_sw_t[i, j]/dividers[z-1], 1)
        dilated_rf[i, j] = np.maximum(reshaped_sw_t[i, j], 1)
        return dilated_rf


class LSTMModel(HelixerModel):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--units', type=int, default=4, help='how many units per LSTM layer')
        self.parser.add_argument('--layers', type=str, default='1', help='how many LSTM layers')
        self.parser.add_argument('--pool-size', type=int, default=10, help='how many bp to predict at once')
        self.parser.add_argument('--dropout', type=float, default=0.0)
        self.parser.add_argument('--layer-normalization', action='store_true')
        self.parser.add_argument('--cpu-compatible', action='store_true',
                                 help='set this to use an LSTM instead of a CuDNNLSTM layer so that the model can run '
                                      'on a CPU if desired. Potentially useful for a quick development test or '
                                      'trouble shooting, but impractically slow for real data.')
        self.parse_args()

        if self.layers.isdigit():
            n_layers = int(self.layers)
            self.layers = [self.units] * n_layers
        else:
            self.layers = eval(self.layers)
            assert isinstance(self.layers, list)
        for key in ["save_model_path", "prediction_output_path", "test_data",
                    "load_model_path", "data_dir"]:
            self.__dict__[key] = self.append_pwd(self.__dict__[key])

    @staticmethod
    def append_pwd(path):
        if path.startswith('/'):
            return path
        else:
            pwd = os.getcwd()
            return '{}/{}'.format(pwd, path)

    def sequence_cls(self):
        return LSTMSequence

    def model(self):
        main_input = Input(shape=(None, self.pool_size * 4), dtype=self.float_precision,
                           name='main_input')
        if self.cpu_compatible:
            LSTMToUse = LSTM
        else:
            LSTMToUse = CuDNNLSTM
        x = Bidirectional(LSTMToUse(self.layers[0], return_sequences=True))(main_input)

        # potential next layers
        if len(self.layers) > 1:
            for layer_units in self.layers[1:]:
                if self.dropout > 0.0:
                    x = Dropout(self.dropout)(x)
                if self.layer_normalization:
                    x = LayerNormalization()(x)
                x = Bidirectional(LSTMToUse(layer_units, return_sequences=True))(x)

        if self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        x = Dense(self.pool_size * 4)(x)
        if self.pool_size > 1:
            x = Reshape((-1, self.pool_size, 4))(x)
        x = Activation('softmax', name='main')(x)

        model = Model(inputs=main_input, outputs=x)
        return model

    def compile_model(self, model):
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        run_metadata = tf.RunMetadata()
        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      sample_weight_mode='temporal',
                      options=run_options,
                      run_metadata=run_metadata)


if __name__ == '__main__':
    model = LSTMModel()
    model.run()
