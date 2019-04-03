# 密集连接：缓解梯度消失问题，加强特征传播，鼓励特征复用，极大的减少了参数量, Dense connection有正则化效果

from keras.models import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten, Concatenate, BatchNormalization, AveragePooling2D, Input, Add, GlobalAveragePooling2D
from keras.utils import plot_model

def Conv2d_BN(x, filters, kernel_size, strides, padding, stage):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=stage+'_conv2d')(x)
    x = BatchNormalization(axis=3, name=stage+'_bn')(x)
    x = Activation('relu', name=stage+'_activation')(x)
    return x

def conv_block(x, growth_rate, stage=None):
    x = Conv2d_BN(x, filters=growth_rate*4, kernel_size=(1, 1), strides=(1, 1), padding='same', stage=stage+'_1_1_block')
    x = Conv2d_BN(x, filters=growth_rate, kernel_size=(3, 3), strides=(1, 1), padding='same', stage=stage+'_3_3_block')
    return x

def transition_block(x, filters, theta=1.0, dropout_rate=None, stage=None):
    x = Conv2d_BN(x, int(filters*theta), (1, 1), (1, 1), 'same', stage)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name=stage+'_Avg_pooling')(x)
    return x

def dense_block(x, layer_cnt, growth_rate, stage):
    concated_feature = x
    for i in range(layer_cnt):
        x = conv_block(concated_feature, growth_rate, stage=stage+'_layer'+str(i+1))
        concated_feature = Concatenate(axis=3)([x, concated_feature])
    return concated_feature

def DenseNet(growth_rate, internal_dense_block=[6, 12, 32, 32], theta=1.0):
    input = Input(shape=(224, 224, 3, ))
    # 112*112, k*2
    x = Conv2d_BN(input, filters=growth_rate*2, kernel_size=(7, 7), strides=(2, 2), padding='same', stage='first_7_7')
    # 56*56, k*2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last')(x)
    x = dense_block(x, internal_dense_block[0], growth_rate, 'Dense_Block_1')

    x = transition_block(x, 256, theta, stage='Transition_Block_1')
    x = dense_block(x, internal_dense_block[1], growth_rate, 'Dense_Block_2')

    x = transition_block(x, 512, theta, stage='Transition_Block_2')
    x = dense_block(x, internal_dense_block[2], growth_rate, 'Dense_Block_3')

    x = transition_block(x, 1280, theta, stage='Transition_Block_3')
    x = dense_block(x, internal_dense_block[3], growth_rate, 'Dense_Block_4')
    
    x = GlobalAveragePooling2D()(x)
    output = Dense(1000, activation='softmax')(x)

    return Model(inputs=input, outputs=output)

model = DenseNet(32, theta=0.5)
plot_model(model, to_file='DenseNet_169_1.png', show_shapes=True, rankdir='TB') 