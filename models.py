import modules
import tensorflow as tf
from tensorflow.keras.layers import *
import layers
from modules import *

########################################################################################################################

########################################################################################################################
def base(n_class, base_filters=160, depth=3):
    def main(inputs):
        filters = [base_filters*i for i in range(1, depth+2)]
        x = inputs
        skip_conn_list = []
        ### Encoder
        for i in range(depth):
            x, skip = encoder.base(filters[i])(x)
            skip = skip_connection.base(filters[i])(skip)
            skip_conn_list.append(skip)
        ### BottleNeck
        x = latent.base(filters[-1])(x)
        ### Decoder
        for i in reversed(range(depth)):
            x = decoder.base(filters[i])(x, skip_conn_list[i])
        x = modules.conv(n_class, 1)(x)
        output = Softmax(-1)(x)
        return output
    return main
