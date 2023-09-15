import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.utils import to_categorical
import numpy as np

model_name = '/home/b/b309233/software/VPRM_preprocessor/neural_network/model02.h5'
################ Custom Generator

class CustomDataGen(keras.utils.Sequence):
    
    def __init__(self, inputs, batch_size, target_gpp=None, target_nee=None,
                 data_range=[0,1], rss=5,
                 target_func = lambda x: x, shuffle=True,):
        normalization_constants = \
         [[0.1053300566011859, 0.10166738198168673],
         [0.292428763712016, 0.0818582429035247],
         [0.08749750285108696, 0.102595429675154],
         [0.10979838649150261, 0.09963923286525665],
         [0.2679398445115099, 0.06124208930412472],
         [0.1688245935203903, 0.05137644148332401],
         [0.0883025578226187, 0.03781302739634237],
         [504178.8208271028, 764651.4971456643],
         [282.69861527151426, 8.366642245686323],
         [0.3306052811511696, 0.07328886144593454],
         [0.32840235210147717, 0.07043098840819884],
         [-6.8508931808647e-05, 0.00010241220780002275],
         [282.8547147449541, 7.482525475965867],
         [282.71946017661816, 6.817477606955349],
         [0, 1]]
        self.batch_size = batch_size
        self.shuffle = shuffle
        for i in range(len(inputs[0])):
            inputs[:, i] = (inputs[:, i] - normalization_constants[i][0])/normalization_constants[i][1]
        self.inputs = inputs
        self.epoch = 0
        self.gpp = target_gpp
        self.nee = target_nee
        tot_len = len(inputs)
        rng = np.random.default_rng(seed=rss)
        if shuffle:
            self.inds = rng.permutation(range(tot_len))[int(data_range[0] * tot_len) : int(data_range[1] * tot_len)]
        else:
            self.inds = np.arange(tot_len)[int(data_range[0] * tot_len) : int(data_range[1] * tot_len)]
        self.n = len(self.inds)
 
    def on_epoch_end(self):
        # print('Epoch {}'.format(self.epoch))
        rng = np.random.default_rng(seed=self.epoch)
        self.epoch += 1
        if self.shuffle:
            self.inds = rng.permutation(self.inds)
    
    def __get_data(self, batch_inds):
        # Generates data containing batch_size samples
        inp1 = np.concatenate([self.inputs[batch_inds, 0:7], 
                               to_categorical(self.inputs[batch_inds, -1],num_classes=8)], axis=1)
       # inp1 = self.inputs[batch_inds, 0:7]
        inp2 = np.array(self.inputs[batch_inds,7:-1])
        # print(np.shape(inp1))
        X_batch =  [inp1, inp2]
        if not self.gpp is None:
            Y_batch = [self.gpp[batch_inds], self.nee[batch_inds]]
        else:
            Y_batch = None
        return X_batch, Y_batch
    
    def __getitem__(self, index):
        if ((index + 1) * self.batch_size > self.n):
            batch_inds = self.inds[index * self.batch_size : self.n]
        else:
            batch_inds = self.inds[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__get_data(batch_inds)
     #   sw = self.sample_weights[batch_inds]
        if y is None:
            return X
        else:
            return X, y
    
    def __len__(self):
        return self.n // self.batch_size + 1 


################ Model Definition

def Dense_Residual(feat_maps_in, feat_maps_out, prev_layer, scale=0.1):
    '''
    A residual unit with dense blocks 
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    id = identitiy_fix_size_dense(feat_maps_in, feat_maps_out, prev_layer)
    dense = dense_block(feat_maps_out, prev_layer)

    x = keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(id)[1:],
               arguments={'scale': scale},)([id, dense])
    return x  # the residual connection


def identitiy_fix_size_dense(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = keras.layers.Dense(feat_maps_out,
                     kernel_regularizer='l1',
                     activation='relu')(prev)
    return prev


def dense_block(feat_maps_out, prev):
    prev = keras.layers.Dense(feat_maps_out,
                 activation='relu',
                 kernel_regularizer='l1')(prev)
    prev = keras.layers.BatchNormalization()(prev)
    prev = keras.layers.Dense(feat_maps_out,
                 activation='relu',
                 kernel_regularizer='l1')(prev)
    prev = keras.layers.BatchNormalization()(prev)
    return prev

def init_model(nlayers=10, nneurons=16):
    input_b1 = keras.layers.Input(
        shape=(15,)) 
    z1 = keras.layers.Dense(nneurons)(input_b1)
    for i in range(nlayers):
        z1 = Dense_Residual(nneurons, nneurons, z1)

    input_b2 = keras.layers.Input(
        shape=(7,))
    z3 =  keras.layers.concatenate([z1, input_b2])
    z3 = keras.layers.Dense(nneurons)(z3)
    for i in range(nlayers):
        z3 = Dense_Residual(nneurons, nneurons, z3)
    z3 = keras.layers.Dense(nneurons)(z3)
    z3 = keras.layers.Dense(nneurons)(z3)

    output_b1 = keras.layers.Dense(1, activation="relu")(z3)
    output_b2 = keras.layers.Dense(1, activation="linear")(z3)
    # The Output
    model= keras.models.Model(inputs=[input_b1, input_b2],
                              outputs=[output_b1, output_b2])
    return model
