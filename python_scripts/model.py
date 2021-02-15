import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import logmelspectr_params as params
import random
import numpy as np

from kapre.composed import get_melspectrogram_layer

checkpoint_path='/nas/home/cborrelli/cnn_bot/checkpoints/vggish/vggish_model.ckpt'


def get_vggish(input_shape, out_dim=2):
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block fc
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1_1')(x)
    x = Dense(4096, activation='relu', name='fc1_2')(x)

    x = Dense(params.EMBEDDING_SIZE, activation='relu', name='fc2')(x)
    x = Dense(out_dim, activation='softmax', name='fc3')(x)

    model = Model(img_input, x, name='vggish')

    # Initialize base model with VGGish weights
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    tensor_layers_list = []
    for key in var_to_shape_map:
        tensor_layers_list.append('/'.join(key.split('/')[:-1]))

    for index, t in enumerate(tensor_layers_list):
        weights_key = t + '/weights'
        bias_key = t + '/biases'
        weights = reader.get_tensor(weights_key)
        biases = reader.get_tensor(bias_key)

        keras_layer_name = t.split('/')[-1]
        if keras_layer_name == 'logits': #or keras_layer_name == 'fc2':
            continue

        model.get_layer(keras_layer_name).set_weights([weights, biases])
    return model


def get_cnn_model(input_shape, out_dim):
    audio_input = Input(shape=input_shape)

    x = get_melspectrogram_layer(input_shape=input_shape, n_fft=512, win_length=512,
                                 hop_length = 256, return_decibel=False,
                                 pad_begin=True, pad_end=True,
                                 sample_rate=16000,
                                 n_mels=128, input_data_format='channels_first',
                                 output_data_format='channels_last',
                                 mel_f_min=0.0, mel_f_max=8000,
                                 )(audio_input)
    # Add noise
    x = GaussianNoise(stddev=np.sqrt(0.1))(x)
    # Normalize
    x = LayerNormalization(axis=-2)(x)

    x = Conv2D(32, strides=(1, 1), kernel_size=(4, 4), activation=None, padding='same', name='conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(48, strides=(1, 1), kernel_size=(5, 5), activation=None, padding='same', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

    x = Conv2D(64, strides=(1, 1), kernel_size=(4, 4), activation=None, padding='same', name='conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    x = Conv2D(128, strides=(1, 1), kernel_size=(4, 2), activation='relu', padding='same', name='conv4')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(out_dim, activation='softmax', name='fc2')(x)

    model = Model(audio_input, x, name='simple_cnn')
    return model


def get_cnn_model_vggish_input(input_shape, out_dim):
    img_input = Input(shape=input_shape)

    x = Conv2D(32, strides=(1, 1), kernel_size=(4, 4), activation=None, padding='same', name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(48, strides=(1, 1), kernel_size=(5, 5), activation=None, padding='same', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

    x = Conv2D(64, strides=(1, 1), kernel_size=(4, 4), activation=None, padding='same', name='conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    x = Conv2D(128, strides=(1, 1), kernel_size=(4, 2), activation='relu', padding='same', name='conv4')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(out_dim, activation='softmax', name='fc2')(x)

    model = Model(img_input, x, name='simple_cnn')
    return model