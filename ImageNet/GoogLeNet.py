# 引入Inception结构
# 中间层的辅助LOSS单元
# 后面的全连接层全部替换为简单的全局平均pooling(Network in Network提出)

from keras.models import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten, Concatenate, BatchNormalization, AveragePooling2D, Input, GlobalAveragePooling2D
from keras.utils import plot_model

def Conv2d_BN(x, filters, kernel_size, strides, padding):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x

def Inception(x, filters):
    branch1_1 = Conv2d_BN(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')

    branch3_3 = Conv2d_BN(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')
    branch3_3 = Conv2d_BN(branch3_3, filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')

    branch5_5 = Conv2d_BN(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')
    branch5_5 = Conv2d_BN(branch5_5, filters=filters, kernel_size=(5, 5), strides=(1, 1), padding='same')

    branch_pooling = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last')(x)
    branch_pooling = Conv2d_BN(branch_pooling, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')

    return Concatenate(axis=3)([branch1_1, branch3_3, branch5_5, branch_pooling])

def GoogLeNet():
    input = Input(shape=(224, 224, 3, ))
    x = Conv2d_BN(input, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Conv2d_BN(x, filters=193, kernel_size=(3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Inception(x, 64)    # 256
    x = Inception(x, 120)   # 480
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)
    x = Inception(x, 208)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    output = Dense(1000, activation='softmax')(x)

    return Model(inputs=input, outputs=output)

model = GoogLeNet()
plot_model(model, to_file='GoogLeNet.png', show_shapes=True, rankdir='TB')