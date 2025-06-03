import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Activation, Lambda
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, AveragePooling3D, concatenate

# ------------------------
# 1) Define MaxwellNet U-Net model
# ------------------------

def build_maxwellnet_unet_3d(input_shape=(64, 64, 64, 1), base_channels=16, depth=4, activation='elu', normalization=0.20):
    inputs = Input(shape=input_shape)
    x = Lambda(lambda z: z / normalization)(inputs)  # Input normalization

    skips = []

    # Encoder
    for d in range(depth):
        channels = base_channels * (2 ** d)
        x = Conv3D(channels, 3, padding='same', kernel_initializer='random_uniform',
                   bias_initializer='zeros')(x)
        x = Activation(activation)(x)
        x = Conv3D(channels, 3, padding='same', kernel_initializer='random_uniform',
                   bias_initializer='zeros')(x)
        skips.append(x)
        x = AveragePooling3D(pool_size=(2, 2, 2))(x)


    # Bottleneck
    channels = base_channels * (2 ** depth)
    x = Conv3D(channels, 3, padding='same', kernel_initializer='random_uniform')(x)
    x = Activation(activation)(x)
    x = Conv3D(channels // 2, 3, padding='same', kernel_initializer='random_uniform')(x)

    # Decoder
    for d in reversed(range(depth)):
        channels = base_channels * (2 ** d)
        x = Conv3DTranspose(channels, 2, strides=2, padding='same', use_bias=False)(x)
        x = concatenate([x, skips[d]], axis=-1)
        x = Conv3D(channels, 3, padding='same', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        x = Activation(activation)(x)
        x = Conv3D(channels // 2, 3, padding='same', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

    outputs = Conv3D(2, (1, 1, 1), use_bias=False, activation=None)(x)
    return Model(inputs=inputs, outputs=outputs, name="MaxwellNet3D")
