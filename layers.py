import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.python.keras.layers.pooling
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *
from keras.layers.convolutional.base_depthwise_conv import DepthwiseConv

import operator
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.layers.pooling import Pooling2D
from keras.layers.convolutional.base_conv import Conv
from keras.utils import tf_utils

import functools
import numpy as np

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
NEW = tf.newaxis
########################################################################################################################

def Adaptive_channel_encoding():
    return


########################################################################################################################
class DeformableConvLayer(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 deformable_groups=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        if deformable_groups is None:
            deformable_groups = filters
        if filters % deformable_groups != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.deformable_groups = deformable_groups

    def build(self, input_shape):
        n_ch = self._get_input_channel(input_shape)
        # kernel_shape = self.kernel_size + (n_ch, self.filters)
        # we want to use depth-wise conv
        # kernel_shape = self.kernel_size + (n_ch // self.groups, self.filters)
        # depthwise_kernel_shape = self.kernel_size + (n_ch, self.depth_multiplier)
        kernel_shape = self.kernel_size + (n_ch * self.filters, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (n_ch, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.built = True

    def get_data_shape(self, inputs):
        input_shape = tensor_shape.TensorShape(inputs.shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_first':
            return input_shape[batch_rank+1:]
        else:
            return input_shape[batch_rank:-1]

    def call(self, inputs, training=None, **kwargs):
        # [bsz, out_h, out_w, kernel_h, * kernel_w * channel_out * 2]
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        offset = tf.nn.conv2d(
            inputs,
            filters=self.offset_layer_kernel,
            strides=[1, *self.strides, 1],
            padding=self.padding.upper(),
            data_format=data_format,
            dilations=[1, *self.dilation_rate, 1]
        )
        offset += self.offset_layer_bias

        inputs = self._pad_input(inputs) if self.padding == 'SAME' else inputs

        # some length
        bsz = inputs.shape[0]
        n_ch = inputs.shape[self._get_channel_axis()]
        in_h, in_w = self.get_data_shape(inputs)
        out_h, out_w = self.get_data_shape(offset)  # output feature map size
        kernel_h, kernel_w = self.kernel_size

        # split offset inot x, y axis
        x_off, y_off = tf.split(offset, 2, axis=self._get_channel_axis())

        # [1, out_h, out_w, filter_h * filter_w]
        x_idx, y_idx = self._get_conv_indices(inputs) # padding을 꼭 VALID로 해야 하나?
        # kernel shape 에 맞추기 위함
        x_idx, y_idx = tf.tile([x_idx, y_idx], [1, bsz, 1, 1, self.deformable_groups])
        x_idx, y_idx = tf.cast([x_idx, y_idx], self.dtype)

        # add offset
        # [out_x, out_y, k^2 * dgroups] + [out_h, out_w, k^2 * dgroups]
        # idx의 padding은 valid인데, off의 padding은 same일 수 있는데 어떻게 맞춰야 하나?
        x_idx, y_idx = x_idx + x_off, y_idx + y_off
        # idx 값이기 때문에 -1
        x_idx = tf.clip_by_value(x_idx, 0, in_w - 1)
        y_idx = tf.clip_by_value(y_idx, 0, in_h - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = tf.cast([x_idx, y_idx], tf.int32)
        y1, x1 = y0 + 1, x0 + 1
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # [out_h, out_w, k^2, n_ch]
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [self._get_pixel_values_at_point(inputs, i) for i in indices]
        x0, x1, y0, y1 = tf.cast([x0, x1, y0, y1], self.dtype)

        # weights
        w0 = (y1 - y_idx) * (x1 - x_idx)
        w1 = (y1 - y_idx) * (x_idx - x0)
        w2 = (y_idx - y0) * (x1 - x_idx)
        w3 = (y_idx - y0) * (x_idx - x0)

        w0, w1, w2, w3 = tf.expand_dims([w0, w1, w2, w3], -1)
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # shaping
        shape = [-1, out_h, out_w, *self.kernel_size, self.deformable_groups, n_ch]
        pixels = tf.reshape(pixels, shape)

        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])

        shape = [-1, out_h * kernel_h, out_w * kernel_w, self.deformable_groups, n_ch]
        pixels = tf.reshape(pixels, shape)

        # copy channels to same group
        # [bsz, out_h, out_w, *kernel_size, deformable_groups, n_ch * feat_in_group]
        feat_in_group = self.filters // self.deformable_groups
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        shape = [-1, out_h * kernel_h, out_w * kernel_w, self.filters * n_ch]
        pixels = tf.reshape(pixels, shape)

        # kernel_shape = [k, k, n_ch * self.filters, 1)]
        outputs = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, *self.kernel_size, 1], 'VALID')
        # add the output feature maps in the same group
        outputs = tf.reshape(outputs, [-1, out_h, out_w, self.filters, n_ch])
        outputs = tf.reduce_sum(outputs, axis=-1)
        if self.use_bias:
            outputs += self.bias
        return self.activation(outputs)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        data_shape = self.get_data_shape(inputs)
        padding_list = []
        for i in range(2):
            filter_size = self.kernel_size[i]
            dilation = self.dilation_rate[i]
            dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
            same_output = (data_shape[i] + self.strides[i] - 1) // self.strides[i]
            valid_output = (data_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
            if same_output == valid_output:
                padding_list += [0, 0]
            else:
                p = dilated_filter_size - 1
                p_0 = p // 2
                padding_list += [p_0, p - p_0]
        if sum(padding_list) != 0:
            padding = [[0, 0],
                       [padding_list[0], padding_list[1]],  # top, bottom padding
                       [padding_list[2], padding_list[3]],  # left, right padding
                       [0, 0]]
            inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, inputs):
        """
        the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = self.get_data_shape(inputs)
        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = x[NEW, :, :, NEW], y[NEW, :, :, NEW]
        x, y = [tf.image.extract_patches(
            i,
            [1, *self.kernel_size, 1],
            [1, *self.strides, 1],
            [1, *self.dilation_rate, 1],
            'VALID' # padding을 꼭 VALID로 해야 하나?
        ) for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return x, y

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.shape #[:4]

        batch_idx = tf.reshape(tf.range(batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

########################################################################################################################
class LAP_2D(Pooling2D):
    """
    LAP: An Attention-Based Module for Faithful Interpretation and Knowledge Injection in Convolutional Neural Networks
        https://arxiv.org/abs/2201.11808
    """
    def __init__(self, pool_size=2, strides=2, padding='VALID', data_format=None, name=None, **kwargs):
        super(LAP_2D, self).__init__(
            self.pool_function, pool_size=pool_size, strides=strides, padding=padding, name=name,
            data_format=data_format, **kwargs
        )
    def build(self, input_shape):
        self.n_ch = input_shape[-1]
        self.alpha = self.add_weight("alpha", shape=[1], initializer=Constant(0.1), constraint=NonNeg())

    def pool_function(self, input, ksize, strides, padding, data_format=None):
        if data_format == 'channels_first':
            input = tf.transpose(input, [0, 2, 3, 1])
        patches = tf.image.extract_patches(input, ksize, strides, [1, 1, 1, 1], padding)
        max = tf.nn.max_pool(input, ksize, strides, padding)
        softmax = tf.math.exp(-(self.alpha ** 2) * (tf.repeat(max, tf.reduce_prod(ksize), -1) - patches) ** 2)
        logit = (softmax * patches + EPSILON)
        # channel에 따라 patch가 섞이므로 분리 후 재 결합
        logit = tf.stack(tf.split(logit, logit.shape[-1] // self.n_ch, -1), -1)
        return tf.reduce_mean(logit, -1)

########################################################################################################################
class AdaPool(Pooling2D):
    """
    AdaPool: Exponential Adaptive Pooling for Information-Retaining Downsampling
        https://arxiv.org/abs/2111.00772
    """
    def __init__(self, pool_size=2, strides=2, padding='VALID', data_format=None, name=None, **kwargs):
        super(AdaPool, self).__init__(
            self.pool_function, pool_size=pool_size, strides=strides, padding=padding, name=name,
            data_format=data_format, **kwargs
        )
    def build(self, input_shape):
        self.n_ch = input_shape[-1]
        self.beta = self.add_weight(
            "beta", shape=[1], initializer=Constant(0.5), constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
        )
    def eMPool(self, x, axis=-1):
        x *= tf.nn.softmax(x, axis)
        return tf.reduce_sum(x, axis)

    def eDSCWPool(self, x, axis=-1):
        DSC = lambda x, x_: tf.math.abs(2 * (x * x_)) / (x ** 2 + x_ ** 2 + EPSILON)
        x_ = tf.reduce_mean(x, axis, keepdims=True)
        dsc = tf.math.exp(DSC(x, x_))
        output = dsc * x / tf.reduce_sum(dsc, axis, keepdims=True)
        return tf.reduce_sum(output, axis)

    def pool_function(self, input, ksize, strides, padding, data_format=None):
        if data_format == 'channels_first':
            input = tf.transpose(input, [0, 2, 3, 1])
        patches = tf.image.extract_patches(input, ksize, strides, [1, 1, 1, 1], padding)
        patches = tf.stack(tf.split(patches, patches.shape[-1] // self.n_ch, -1), -1)
        return self.eMPool(patches) * self.beta + self.eDSCWPool(patches) * (1 - self.beta)

########################################################################################################################
class SaBN(Layer):
    """
    Sandwich Batch Normalization: A Drop-In Replacement for Feature Distribution Heterogeneity
        https://arxiv.org/abs/2102.11382
    """
    def __init__(self, n_class, axis=-1):
        super(SaBN, self).__init__()
        self.n_class = n_class
        self.BN = BatchNormalization(axis=axis)
        self.axis = [axis] if isinstance(axis, int) else axis

    def build(self, input_shape):
        param_shape = self.get_param_shape(input_shape)

        self.scale = self.add_weight("scale", shape=param_shape, initializer='ones')
        self.offset = self.add_weight("offset", shape=param_shape, initializer='zeros')

    def get_param_shape(self, input_shape):
        ndims = len(input_shape)
        # negative parameter to positive parameter
        axis = [ndims + ax if ax < 0 else ax for ax in self.axis]
        axis_to_dim = {x: input_shape[x] for x in axis}
        param_shape = [axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)]
        param_shape = [self.n_class] + param_shape
        print(f'param_shape = {param_shape}')
        return param_shape

    def get_slice(self, x, label):
        x = tf.gather_nd(x, label)
        x = tf.squeeze(x, 1)

        return x

    def call(self, inputs, label, training=None, **kargs):
        if training == False:
            return self.BN(inputs, training=training)
        output = self.BN(inputs, training=training)

        # label = tf.argmax(label, -1)
        scale = self.get_slice(self.scale, label)
        offset = self.get_slice(self.offset, label)
        print(f'label = {label}')
        print(f'offset = {self.offset.shape}')
        print(f'offset_gather = {offset.shape}')
        output = scale * output + offset
        print(f'output = {output.shape}')
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_class': self.n_class,
        })
        return config

########################################################################################################################
class sep_bias(Layer):
    def __init__(self, input_dims):
        super(sep_bias, self).__init__()
        assert input_dims > 0
        self.input_dims = input_dims

    def build(self, input_shape):
        self.scale = Embedding(self.input_dims, input_shape[-1], embeddings_initializer='ones')
        self.offset = Embedding(self.input_dims, input_shape[-1], embeddings_initializer='zeros')

    def call(self, inputs, label=0, training=None):
        assert self.input_dims >= label
        x = self.scale(label) * inputs + self.offset(label)
        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.input_dims,
        })
        return config

########################################################################################################################
class Attention(Layer):
    def __init__(self, filters, axis=-1):
        super(Attention, self).__init__()
        self.filters = filters
        self.axis = [axis] if isinstance(axis, int) else axis

    def build(self, input_shape):
        n_ch = input_shape[-1]

        self.scale = self.add_weight("scale", shape=[n_ch, self.units], initializer='HeNormal')
        self.offset = self.add_weight("offset", shape=[self.filters, n_ch], initializer='HeNormal')

    def call(self, inputs, training=None, **kargs):
        tf.matmul(a=inputs, b=self.kernel)


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "scale": self.scale,
            "offset": self.offset,
        })

########################################################################################################################
class DepthwiseConv3D(DepthwiseConv):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='same',
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv3D, self).__init__(
            3,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        outputs = backend.conv3d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0],
                                             self.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1],
                                             self.dilation_rate[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)


########################################################################################################################
class flat(Layer):
    """
    flat inputs except channel dimension.
    It function like layers.Flatten except channel dimension

    example:
    x = tf.constant([1, 3, 3, 2])
    flat()(x).shape
    # >>> [1, 9, 2]
    """
    def __init__(self, data_format=None, **kwargs):
        super(flat, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(min_ndim=1)
        self._channels_first = self.data_format == 'channels_first'

    def call(self, inputs, **kwargs):
        if self._channels_first:
            rank = inputs.shape.rank
            if rank and rank > 1:
                # Switch to channels-last format.
                permutation = [0]
                permutation.extend(range(2, rank))
                permutation.append(1)
                inputs = tf.transpose(inputs, perm=permutation)

        if tf.executing_eagerly():
            flattened_shape = tf.constant([inputs.shape[0], -1, inputs.shape[-1]])
            return tf.reshape(inputs, flattened_shape)
        else:
            input_shape = inputs.shape
            rank = input_shape.rank
            if rank == 1:
                return tf.expand_dims(inputs, axis=1)
            else:
                batch_dim = tf.compat.dimension_value(input_shape[0])
                non_batch_dims = input_shape[1:]
                # Reshape in a way that preserves as much shape info as possible.
                if non_batch_dims.is_fully_defined():
                    last_dim = int(functools.reduce(operator.mul, non_batch_dims[:-1]))
                    flattened_shape = tf.constant([-1, last_dim, non_batch_dims[-1]])
                elif batch_dim is not None:
                    flattened_shape = tf.constant([int(batch_dim), -1, input_shape[-1]])
                else:
                    flattened_shape = [tf.shape(inputs)[0], -1, tf.shape(inputs)[-1]]
                return tf.reshape(inputs, flattened_shape)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if not input_shape:
            output_shape = tf.TensorShape([1])
        else:
            output_shape = [input_shape[0]]
        if np.all(input_shape[1:]):
            output_shape += [np.prod(input_shape[1:-1], dtype=int)]
        else:
            output_shape += [None]
        output_shape += input_shape[-1]
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super(flat, self).get_config()
        config.update({'data_format': self.data_format})
        return config

########################################################################################################################
class global_attention(Layer):
    """
    Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions
        https://arxiv.org/abs/2112.05561
    :return:
    """
    def __init__(self, kernel=7, groups=1, squeeze_rate=0.7):
        super(global_attention, self).__init__()
        self.squeeze_rate = squeeze_rate
        self.kernel = kernel
        self.groups = groups

    def build(self, input_shape):
        rank = tf.rank(input_shape)
        self.GAP = GlobalAveragePooling2D if rank == 4 else GlobalAveragePooling3D
        self.GMP = GlobalMaxPooling2D if rank == 4 else GlobalMaxPooling3D

        self.activation = tf.nn.relu

        self.squeeze = Dense(int(input_shape[-1] * self.squeeze_rate))
        self.extend = Dense(input_shape[-1], activation='relu')

        conv = Conv2D if rank == 4 else Conv3D
        self.spatial_convolution = conv(1, self.kernel, padding='same', groups=self.groups)

    def channel_attention(self, x):
        def main(x):
            x = self.squeeze(x)
            x = self.extend(x)
            x = self.activation(x)
            return x
        GAP = main(self.GAP()(x))
        GMP = main(self.GMP()(x))

        return tf.nn.sigmoid(GAP + GMP)

    def spatial_attention(self, x):
        GAP = tf.reduce_mean(x[0], -1)
        GMP = tf.reduce_max(x[1], -1)

        spatial_atttention = self.spatial_convolution(tf.concat([GAP, GMP], -1))
        spatial_atttention = tf.nn.sigmoid(spatial_atttention)
        return spatial_atttention

    def call(self, inputs, label=0, training=None):
        channel_attention = self.channel_attention(inputs)
        x = inputs * channel_attention

        spatial_attention = self.spatial_attention(x)
        x *= spatial_attention

        return x + inputs

########################################################################################################################
# class LI_2D(tf.python.keras.layers.convolutional.Conv2D):
#     """
#     Dilated Convolutions with Lateral Inhibitions for Semantic Image Segmentation
#         https://arxiv.org/abs/2006.03708
#     """
#     def __init__(self,
#                  filters,
#                  kernel_size=3,
#                  strides=1,
#                  padding='valid',
#                  data_format=None,
#                  dilation_rate=1,
#                  groups=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer='zeros',
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  intensity=0.2,
#                  **kwargs):
#         super(LI_2D, self).__init__(
#             filters=filters,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_format=data_format,
#             dilation_rate=dilation_rate,
#             groups=groups,
#             activation=activations.get(activation),
#             use_bias=use_bias,
#             kernel_initializer=initializers.get(kernel_initializer),
#             bias_initializer=initializers.get(bias_initializer),
#             kernel_regularizer=regularizers.get(kernel_regularizer),
#             bias_regularizer=regularizers.get(bias_regularizer),
#             activity_regularizer=regularizers.get(activity_regularizer),
#             kernel_constraint=constraints.get(kernel_constraint),
#             bias_constraint=constraints.get(bias_constraint),
#             **kwargs
#         )
#         self.intensity = intensity
#
#     def build(self, input_shape):
#         input_shape = tensor_shape.TensorShape(input_shape)
#         if len(input_shape) != 4:
#             raise ValueError('Inputs should have rank 4. Received input '
#                              'shape: ' + str(input_shape))
#         input_channel = self._get_input_channel(input_shape)
#         if input_channel % self.groups != 0:
#             raise ValueError(
#                 'The number of input channels must be evenly divisible by the number '
#                 'of groups. Received groups={}, but the input has {} channels '
#                 '(full input shape is {}).'.format(self.groups, input_channel,
#                                                    input_shape))
#         kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)
#
#         self.weight = self.add_weight("weight", shape=input_channel, initializer='HeNormal')
#
#         channel_axis = self._get_channel_axis()
#         if input_shape.dims[channel_axis].value is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         input_dim = int(input_shape[channel_axis])
#         self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
#
#         self.kernel = self.add_weight(
#             name='kernel',
#             shape=kernel_shape,
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#             trainable=True,
#             dtype=self.dtype)
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 name='bias',
#                 shape=(self.filters,),
#                 initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
#                 trainable=True,
#                 dtype=self.dtype)
#         else:
#             self.bias = None
#         self.built = True
#
#     def distance_factor(self, size=3, factor=1.5):
#         """ Euclidean distance """
#         x = np.mgrid[:size, :size]
#         x = (x - size // 2) * 1.0
#         x = (x**2).sum(0) + factor
#         x = tf.Variable(1/x, dtype='float32')
#         return x
#
#     def add_bias(self, outputs):
#         output_rank = outputs.shape.rank
#         if self.rank == 1 and self._channels_first:
#             # nn.bias_add does not accept a 1D input tensor.
#             bias = array_ops.reshape(self.bias, (1, self.filters, 1))
#             outputs += bias
#         else:
#             # Handle multiple batch dimensions.
#             if output_rank is not None and output_rank > 2 + self.rank:
#                 def _apply_fn(o):
#                     return nn.bias_add(o, self.bias, data_format=self._tf_data_format)
#
#                 outputs = conv_utils.squeeze_batch_dims(
#                     outputs, _apply_fn, inner_rank=self.rank + 1)
#             else:
#                 outputs = nn.bias_add(
#                     outputs, self.bias, data_format=self._tf_data_format)
#         return outputs
#
#     def reconstruction(self):
#         return
#
#     def call(self, inputs, label=0, training=None):
#         patches = tf.image.extract_patches(
#             inputs,
#             [1, self.kernel_size, self.kernel_size, 1],
#             [1, self.strides, self.strides, 1],
#             [1, self.dilation_rate, self.dilation_rate, 1],
#             padding='VALID',
#         ) # B, n_patch, n_patch, patch_size(3*3)
#         distance_factor = self.distance_factor(self.kernel_size)
#         distance_factor = tf.reshape(distance_factor, [1, 1, 1, -1]) # [k, k] > [1, 1, 1, patch_size]
#         # patch encoding
#         patches *= distance_factor
#
#         outputs = self._convolution_op(inputs, self.kernel)
#
#         if self.use_bias:
#             outputs = self.add_bias(outputs)
#
#
#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "weight": self.weight,
#         })
#         return config

########################################################################################################################
class channel_attention_base(Layer):
    def __init__(self, squeeze_rate=0.6):
        super(channel_attention_base, self).__init__()
        self.squeeze_rate = squeeze_rate
        self.BN1 = BatchNormalization()
        self.BN2 = BatchNormalization()

    def build(self, input_shape):
        self.n_channel = input_shape[-1]
        rank = len(input_shape)
        self.GAP = GlobalAveragePooling2D(keepdims=True) if rank == 4 \
            else GlobalAveragePooling3D(keepdims=True)
        self.squeezed_dense = Dense(self.squeeze_rate * self.n_channel)
        self.released_dense = Dense(self.n_channel)

class SE(channel_attention_base):
    """
    Squeeze-and-Excitation Networks
        https://arxiv.org/abs/1709.01507
    """
    def __init__(self, squeeze_rate=0.6):
        super(SE, self).__init__()
        self.squeeze_rate = squeeze_rate

    def call(self, inputs, training=None, **kargs):
        x = self.GAP(inputs)
        x = self.squeezed_dense(x)
        x = self.BN1(x)
        x = tf.nn.relu(x)

        x = self.released_dense(x)
        x = self.BN2(x)
        x = tf.math.sigmoid(x)
        return inputs * x

class SB(channel_attention_base):
    """
    Shift-and-Balance Attention
        https://arxiv.org/abs/2103.13080
    """
    def __init__(self, squeeze_rate=0.6):
        super(SB, self).__init__()
        self.squeeze_rate = squeeze_rate

    def call(self, inputs, training=None, **kargs):
        x = self.GAP(inputs)
        x = self.squeezed_dense(x)
        x = self.BN1(x)
        x = tf.nn.relu(x)

        x = self.released_dense(x)
        x = self.BN2(x)
        x = tf.math.tanh(x)
        return inputs + x


########################################################################################################################
########################################################################################################################
# class Conv(Layer):
#     def __init__(self,
#                  rank,
#                  filters,
#                  kernel_size,
#                  strides=1,
#                  padding='valid',
#                  data_format=None,
#                  dilation_rate=1,
#                  groups=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  trainable=True,
#                  name=None,
#                  conv_op=None,
#                  **kwargs):
#         super(Conv, self).__init__(trainable=trainable, name=name,
#                                    activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
#         self.rank = rank
#         if isinstance(filters, float):
#             filters = int(filters)
#         if filters is not None and filters < 0:
#             raise ValueError(f'Received a negative value for `filters`.'
#                              f'Was expecting a positive value, got {filters}.')
#         self.filters = filters
#         self.groups = groups or 1
#         self.kernel_size = conv_utils.normalize_tuple(
#             kernel_size, rank, 'kernel_size')
#         self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
#         self.padding = conv_utils.normalize_padding(padding)
#         self.data_format = conv_utils.normalize_data_format(data_format)
#         self.dilation_rate = conv_utils.normalize_tuple(
#             dilation_rate, rank, 'dilation_rate')
#
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.input_spec = InputSpec(min_ndim=self.rank + 2)
#
#         self._validate_init()
#         self._is_causal = self.padding == 'causal'
#         self._channels_first = self.data_format == 'channels_first'
#         self._tf_data_format = conv_utils.convert_data_format(
#             self.data_format, self.rank + 2)
#
#     def _validate_init(self):
#         if self.filters is not None and self.filters % self.groups != 0:
#             raise ValueError(
#                 'The number of filters must be evenly divisible by the number of '
#                 'groups. Received: groups={}, filters={}'.format(
#                     self.groups, self.filters))
#
#         if not all(self.kernel_size):
#             raise ValueError('The argument `kernel_size` cannot contain 0(s). '
#                              'Received: %s' % (self.kernel_size,))
#
#         if not all(self.strides):
#             raise ValueError('The argument `strides` cannot contains 0(s). '
#                              'Received: %s' % (self.strides,))
#
#         if (self.padding == 'causal' and not isinstance(self,
#                                                         (Conv1D, SeparableConv1D))):
#             raise ValueError('Causal padding is only supported for `Conv1D`'
#                              'and `SeparableConv1D`.')
#
#     def build(self, input_shape):
#         input_shape = tensor_shape.TensorShape(input_shape)
#         input_channel = self._get_input_channel(input_shape)
#         if input_channel % self.groups != 0:
#             raise ValueError(
#                 'The number of input channels must be evenly divisible by the number '
#                 'of groups. Received groups={}, but the input has {} channels '
#                 '(full input shape is {}).'.format(self.groups, input_channel,
#                                                    input_shape))
#         kernel_shape = self.kernel_size + (input_channel // self.groups,
#                                            self.filters)
#
#         self.kernel = self.add_weight(
#             name='kernel',
#             shape=kernel_shape,
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#             trainable=True,
#             dtype=self.dtype)
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 name='bias',
#                 shape=(self.filters,),
#                 initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
#                 trainable=True,
#                 dtype=self.dtype)
#         else:
#             self.bias = None
#         channel_axis = self._get_channel_axis()
#         self.input_spec = InputSpec(min_ndim=self.rank + 2,
#                                     axes={channel_axis: input_channel})
#
#         # Convert Keras formats to TF native formats.
#         if self.padding == 'causal':
#             tf_padding = 'VALID'  # Causal padding handled in `call`.
#         elif isinstance(self.padding, str):
#             tf_padding = self.padding.upper()
#         else:
#             tf_padding = self.padding
#         tf_dilations = list(self.dilation_rate)
#         tf_strides = list(self.strides)
#
#         tf_op_name = self.__class__.__name__
#         if tf_op_name == 'Conv1D':
#             tf_op_name = 'conv1d'  # Backwards compat.
#
#         self._convolution_op = functools.partial(
#             nn_ops.convolution_v2,
#             strides=tf_strides,
#             padding=tf_padding,
#             dilations=tf_dilations,
#             data_format=self._tf_data_format,
#             name=tf_op_name)
#         self.built = True
#
#     def call(self, inputs):
#         input_shape = inputs.shape
#
#         if self._is_causal:  # Apply causal padding to inputs for Conv1D.
#             inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
#
#         outputs = self._convolution_op(inputs, self.kernel)
#
#         if self.use_bias:
#             output_rank = outputs.shape.rank
#             if self.rank == 1 and self._channels_first:
#                 # nn.bias_add does not accept a 1D input tensor.
#                 bias = array_ops.reshape(self.bias, (1, self.filters, 1))
#                 outputs += bias
#             else:
#                 # Handle multiple batch dimensions.
#                 if output_rank is not None and output_rank > 2 + self.rank:
#
#                     def _apply_fn(o):
#                         return nn.bias_add(o, self.bias, data_format=self._tf_data_format)
#
#                     outputs = conv_utils.squeeze_batch_dims(
#                         outputs, _apply_fn, inner_rank=self.rank + 1)
#                 else:
#                     outputs = nn.bias_add(
#                         outputs, self.bias, data_format=self._tf_data_format)
#
#         if not context.executing_eagerly():
#             # Infer the static output shape:
#             out_shape = self.compute_output_shape(input_shape)
#             outputs.set_shape(out_shape)
#
#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs
#
#     def _spatial_output_shape(self, spatial_input_shape):
#         return [
#             conv_utils.conv_output_length(
#                 length,
#                 self.kernel_size[i],
#                 padding=self.padding,
#                 stride=self.strides[i],
#                 dilation=self.dilation_rate[i])
#             for i, length in enumerate(spatial_input_shape)
#         ]
#
#     def compute_output_shape(self, input_shape):
#         input_shape = tensor_shape.TensorShape(input_shape).as_list()
#         batch_rank = len(input_shape) - self.rank - 1
#         if self.data_format == 'channels_last':
#             return tensor_shape.TensorShape(
#                 input_shape[:batch_rank]
#                 + self._spatial_output_shape(input_shape[batch_rank:-1])
#                 + [self.filters])
#         else:
#             return tensor_shape.TensorShape(
#                 input_shape[:batch_rank] + [self.filters] +
#                 self._spatial_output_shape(input_shape[batch_rank + 1:]))
#
#     def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
#         return False
#
#     def get_config(self):
#         config = {
#             'filters':
#                 self.filters,
#             'kernel_size':
#                 self.kernel_size,
#             'strides':
#                 self.strides,
#             'padding':
#                 self.padding,
#             'data_format':
#                 self.data_format,
#             'dilation_rate':
#                 self.dilation_rate,
#             'groups':
#                 self.groups,
#             'activation':
#                 activations.serialize(self.activation),
#             'use_bias':
#                 self.use_bias,
#             'kernel_initializer':
#                 initializers.serialize(self.kernel_initializer),
#             'bias_initializer':
#                 initializers.serialize(self.bias_initializer),
#             'kernel_regularizer':
#                 regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer':
#                 regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer':
#                 regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint':
#                 constraints.serialize(self.kernel_constraint),
#             'bias_constraint':
#                 constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Conv, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def _compute_causal_padding(self, inputs):
#         """Calculates padding for 'causal' option for 1-d conv layers."""
#         left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
#         if getattr(inputs.shape, 'ndims', None) is None:
#             batch_rank = 1
#         else:
#             batch_rank = len(inputs.shape) - 2
#         if self.data_format == 'channels_last':
#             causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
#         else:
#             causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
#         return causal_padding
#
#     def _get_channel_axis(self):
#         if self.data_format == 'channels_first':
#             return -1 - self.rank
#         else:
#             return -1
#
#     def _get_input_channel(self, input_shape):
#         channel_axis = self._get_channel_axis()
#         if input_shape.dims[channel_axis].value is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         return int(input_shape[channel_axis])
#
#     def _get_padding_op(self):
#         if self.padding == 'causal':
#             op_padding = 'valid'
#         else:
#             op_padding = self.padding
#         if not isinstance(op_padding, (list, tuple)):
#             op_padding = op_padding.upper()
#         return op_padding
