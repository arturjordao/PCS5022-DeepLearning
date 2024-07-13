#https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
from keras import backend
from keras import layers
from keras import models
from keras.layers import *
import numpy as np
from sklearn.utils import gen_batches
from tensorflow.keras.applications.resnet50 import preprocess_input

def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def resnet(input_shape=(224, 224, 3), blocks=None, num_classes=1000):
    stacks = len(blocks)
    num_filters = 64
    bn_axis = 3

    img_input = Input(input_shape)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = Conv2D(num_filters, 7, strides=2, use_bias=True, name='conv1_conv')(x)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    for stage in range(0, stacks):
        num_res_blocks = blocks[stage]
        name = 'conv'+str(stage+2)
        if stage == 0:
            stride = 1
        else:
            stride = 2
        x = block1(x, filters=num_filters, stride=stride, name=name+'_block1')

        for res_block in range(2, num_res_blocks+1):
            x = block1(x, num_filters, conv_shortcut=False, name=name+'_block'+str(res_block))

        num_filters = num_filters * 2

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation='softmax', name='probs')(x)
    inputs = img_input

    model = models.Model(inputs, x, name='ResNetBN')
    return model

def count_res_blocks(model):
    res_blocks = {}

    for layer in model.layers:
        if isinstance(layer, layers.Add):
            dim = layer.output_shape[1]#1 and 2 are the spatial dimensions
            res_blocks[dim] = res_blocks.get(dim, 0) + 1

    return list(res_blocks.values())

if __name__ == '__main__':
    np.random.seed(12227)

    input_shape = (224, 224, 3)
    num_classes = 1000

    blocks = [3, 4, 6, 3]#ResNet50
    #blocks = [3, 4, 23, 3]#ResNet101
    # blocks = [3, 8, 36, 3]#ResNet152

    model = resnet(input_shape=input_shape,
                   blocks=blocks,
                   num_classes=num_classes)
    model.load_weights('<Path to the downloaded weights>')
    
    model.summary()
    
    X = np.random.rand(30, 224, 224, 3)
    model.predict(X)