# 层数非常深，已经超过百层
# 引入残差单元来解决退化问题，学习F(x)=H(x)-x
# 考虑到x的维度与F(X)维度可能不匹配情况，需进行维度匹配：
# （1）zero_padding:对恒等层进行0填充的方式将维度补充完整。这种方法不会增加额外的参数
# （2）projection:在恒等层采用1x1的卷积核来增加维度。这种方法会增加额外的参数
# 文中还有第三种方法，但通过实验发现第三种方法会使performance急剧下降，故不采用
 

from keras.models import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten, Concatenate, BatchNormalization, AveragePooling2D, Input, Add, GlobalAveragePooling2D
from keras.utils import plot_model

def Conv2d_BN(x, filters, kernel_size, strides, padding):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    return x

def Conv_Block(input_x, filters, with_projection=False):
    x = Conv2d_BN(input_x, filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='same')
    x = Conv2d_BN(x, filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding='same')
    x = Conv2d_BN(x, filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='same')
    if with_projection:
        input_x = Conv2d_BN(input_x, filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='same')
    x = Add()([x, input_x])
    return x

def ResNet_50():
    input = Input(shape=(224, 224, 3, ))
    # 112*112, 64
    x = Conv2d_BN(input, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    # 56*56, 256
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Conv_Block(x, [64, 64, 256], with_projection=True)
    x = Conv_Block(x, [64, 64, 256], with_projection=False)
    x = Conv_Block(x, [64, 64, 256], with_projection=False)
    # 28*28, 512
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Conv_Block(x, [128, 128, 512], with_projection=True)
    x = Conv_Block(x, [128, 128, 512], with_projection=False)
    x = Conv_Block(x, [128, 128, 512], with_projection=False)
    x = Conv_Block(x, [128, 128, 512], with_projection=False)
    # 14*14, 1024
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Conv_Block(x, [256, 256, 1024], with_projection=True)
    x = Conv_Block(x, [256, 256, 1024], with_projection=False)
    x = Conv_Block(x, [256, 256, 1024], with_projection=False)
    x = Conv_Block(x, [256, 256, 1024], with_projection=False)
    x = Conv_Block(x, [256, 256, 1024], with_projection=False)
    x = Conv_Block(x, [256, 256, 1024], with_projection=False)
    # 7*7, 2048
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = Conv_Block(x, [512, 512, 2048], with_projection=True)
    x = Conv_Block(x, [512, 512, 2048], with_projection=False)
    x = Conv_Block(x, [512, 512, 2048], with_projection=False)
    # 1*1, 2048
    x = GlobalAveragePooling2D()(x)
    output = Dense(1000, activation='softmax')(x)

    return Model(inputs=input, outputs=output)

model = ResNet_50()
plot_model(model, to_file='ResNet_50.png', show_shapes=True, rankdir='TB')