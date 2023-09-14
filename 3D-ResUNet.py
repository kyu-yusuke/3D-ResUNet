from tensorflow.keras import backend 
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate, Activation, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_pred_f * y_true_f)
    return (2.0 * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def first_res_block(input_layer, channel, kernel, activation):
    conv01 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(1,1,1))(input_layer)
    bn01 = InstanceNormalization()(conv01)
    act01 = Activation(activation=activation)(bn01)
    conv02 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(1,1,1))(act01)

    shcut01 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(1,1,1))(input_layer)
    bn03 = InstanceNormalization()(shcut01)
    res01 = add([bn03, conv02])

    return res01

def encode_res_block(input_layer, channel, kernel, activation):
    bn01 = InstanceNormalization()(input_layer)
    act01 = Activation(activation=activation)(bn01)
    conv01 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(2,2,2))(act01)
    bn02 = InstanceNormalization()(conv01)
    act02 = Activation(activation=activation)(bn02)
    conv02 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(1,1,1))(act02)

    shcut01 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(2,2,2))(input_layer)
    bn03 = InstanceNormalization()(shcut01)
    res01 = add([bn03, conv02])

    return res01

def decode_res_block(input_layer, channel, kernel, activation):
    bn01 = InstanceNormalization()(input_layer)
    act01 = Activation(activation=activation)(bn01)
    conv01 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(1,1,1))(act01)
    bn02 = InstanceNormalization()(conv01)
    act02 = Activation(activation=activation)(bn02)
    conv02 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(1,1,1))(act02)

    shcut01 = Conv3D(filters=channel, kernel_size=kernel, padding='same', strides=(1,1,1))(input_layer)
    bn03 = InstanceNormalization()(shcut01)
    res01 = add([bn03, conv02])

    return res01

# Main function
def UNet(input_shape, kernel, act_method, ini_channel, layer_depth, learning_rate):
    # Encoding 
    encod_ch = [ini_channel * 2 ** i for i in range(layer_depth)]
    encod_addlayer = []

    initial_input = Input(shape=input_shape)

    for i in range(layer_depth):
        if i == 0:
            input_layer = initial_input
            add_layer = first_res_block(input_layer=input_layer, channel=encod_ch[i], kernel=kernel, activation=act_method)
            encod_addlayer.append(add_layer)
        else:
            input_layer = encod_addlayer[i-1]
            add_layer = encode_res_block(input_layer=input_layer, channel=encod_ch[i], kernel=kernel, activation=act_method)
            encod_addlayer.append(add_layer) 

    # Deepest Layer
    input_layer = encod_addlayer[-1]
    deepest_ch = encod_ch[-1] * 2
    deepest_addlayer = encode_res_block(input_layer=input_layer, channel=deepest_ch, kernel=kernel, activation=act_method)

    # Decoding
    decod_ch = list(reversed(encod_ch))
    decod_addlayer = []

    for i in range(layer_depth):
        if i == 0:
            input_layer = deepest_addlayer
        else:
            input_layer = decod_addlayer[i-1]
        
        up_ = UpSampling3D(size=(2,2,2), data_format='channels_last')(input_layer)
        concat_ = Concatenate()([up_, encod_addlayer[-1-i]])
        add_layer = decode_res_block(input_layer=concat_, channel=decod_ch[i], kernel=kernel, activation=act_method)
        decod_addlayer.append(add_layer)
            
    # Last conv
    output_layer = Conv3D(filters=1, kernel_size=(1,1,1), activation='sigmoid', padding='same')(decod_addlayer[-1])

    # Model
    model = Model(inputs=initial_input, outputs=output_layer)
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

    return model