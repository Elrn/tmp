"""
dwi : contain skull
adc
seg

! 기본적으로 float64 로 되어있어 변환 필요

-------------------------------------------------
원 이미지는 channel dimension 을 포함하지 않음
> parsing 에서 dims 추가 필요 !
-------------------------------------------------

"""
########################################################################################################################
import tensorflow as tf
import utils
import os
from os.path import basename
import numpy as np
import nibabel as nib
import cv2

import flags
FLAGS = flags.FLAGS

num_class = 2
input_shape = [128, 128, 1] # Z: 200

########################################################################################################################
""" parse function """
########################################################################################################################
def parse_fn(data, seg): # RANK: (4, 3)
    data = tf.expand_dims(data, -1)
    data = tf.image.per_image_standardization(data)
    data = tf.cast(data, 'float32')
    seg = tf.cast(seg, 'int32')
    seg = tf.one_hot(seg, num_class, axis=-1)

    return (data, seg)

def pred_parse_fn(x):
    x = tf.expand_dims(x, -1)
    x = tf.image.per_image_standardization(x)
    # x = tf.image.resize(x, input_shape[:2])
    x = tf.cast(x, 'float32')
    return x

########################################################################################################################
""" utils """
########################################################################################################################
def validation_split_fn(dataset, validation_split):
    len_dataset = tf.data.experimental.cardinality(dataset).numpy()
    valid_count = int(len_dataset * validation_split)
    print(f'[Dataset|split] Total: "{len_dataset}", Train: "{len_dataset-valid_count}", Valid: "{valid_count}"')
    return dataset.skip(valid_count), dataset.take(valid_count)

########################################################################################################################
""" Build """
########################################################################################################################
def build_for_pred(paths):
    def get_dataset(x):
        return tf.data.Dataset.from_tensor_slices(
            x  # num_files, z, x, y
        ).cache(
        ).map(
            map_func=pred_parse_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        # ).unbatch(
        ).batch(batch_size=FLAGS.bsz,
        )

    arrs, depths, headers, affines, img_shapes = [], [], [], [], []
    for path in paths:
        data = nib.load(path)
        header = data.header
        img_sz = header.get_data_shape()[:2]
        affine = data.affine
        ndarray = data.get_fdata()
        ndarray = cv2.resize(ndarray, dsize=input_shape[:2], interpolation=cv2.INTER_CUBIC)
        ndarray = np.transpose(ndarray, [2, 0, 1])
        depth = ndarray.shape[0]

        arrs.append(ndarray)
        depths.append(depth)
        headers.append(header)
        affines.append(affine)
        img_shapes.append(img_sz)

    arrs = np.concatenate(arrs, 0)
    dataset = get_dataset(arrs)
    return dataset, depths, headers, affines, img_shapes

########################################################################################################################
def build(batch_size, validation_split=0.1):
    assert 0 <= validation_split <= 0.5
    file_path = 'C:\\dataset\\dwi_image_preprocessed.npz'
    print(f'[Dataset] load:"{basename(file_path)}", batch size:"{batch_size}", split:"{validation_split}"')
    with np.load(file_path) as data: # x, y, z
        adc = tf.concat([data['adc_lesion_x'], data['adc_nolesion_x']], -1)
        dwi = tf.concat([data['dwi_lesion_x'], data['dwi_nolesion_x']], -1)
        d = tf.concat([adc, dwi], -1)
        seg = tf.concat([data['dwi_lesion_y'], data['dwi_nolesion_y']], -1)
        seg = tf.concat([seg, seg], -1)
        d = tf.transpose(d, [2, 0, 1])
        seg = tf.transpose(seg, [2, 0, 1])
        dataset = load((d, seg), batch_size)
        if validation_split != None and validation_split > 0.:
            return validation_split_fn(dataset, validation_split)
        else:
            return dataset, None

def load(data, batch_size, drop=True):
    return tf.data.Dataset.from_tensor_slices(
        data
    # ).prefetch(
    #     tf.data.experimental.AUTOTUNE
    # ).interleave(
    #     lambda x : tf.data.Dataset(x).map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
    #     cycle_length = tf.data.experimental.AUTOTUNE,
    #     num_parallel_calls = tf.data.experimental.AUTOTUNE
    # ).repeat(
    #     count=3
    ).shuffle(
        1048,
    #     reshuffle_each_iteration=True
    # ).cache(
    ).map(
        map_func=parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    # ).unbatch( # batch > unbatch > batch 시 cardinality = -2 로 설정됨
    ).batch(
        batch_size=batch_size,
        drop_remainder=drop,
    )

########################################################################################################################
""" for plot """
########################################################################################################################
def build_test(batch_size):
    # [25, 128, 128]
    file_path = 'C:\\dataset\\KUMC\\plot_CHW.npz'
    with np.load(file_path) as data:
        dwi, adc, seg = data['dwi'], data['adc'], data['seg']

    dataset_dwi = load_test((dwi, seg), batch_size)
    dataset_adc = load_test((adc, seg), batch_size)
    return dataset_adc, dataset_dwi

def load_test(data, batch_size, drop=True):
    return tf.data.Dataset.from_tensors(data
        ).map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).unbatch(
        ).batch(batch_size=batch_size, drop_remainder=drop,
        )


########################################################################################################################
""" Post-processing """
########################################################################################################################
def post_processing(outputs, depths, headers, affines, img_shapes):
    """
    2D slices 로 진행했기 때문에 unbatch 된 output을 원래 크기에 맞도록 분할하여 stack으로 쌓아야 한다.

    transpose 여부 확인 필요 !
    """
    def upsampling(img, shape):
        slices = []
        for slice in img:
            resized_slice = cv2.resize(slice, dsize=shape, interpolation=cv2.INTER_CUBIC)
            slices.append(resized_slice)
        return np.stack(slices, 0)

    outputs = np.squeeze(np.argmax(outputs, -1)) # one_hot >> indices
    sep_inputs = []
    for depth, img_sz in zip(depths, img_shapes):
        x = outputs[:depth]
        x = np.transpose(x, [1, 2, 0])
        x = x.astype('uint8') # for cv2 resize
        x = cv2.resize(x, dsize=img_sz, interpolation=cv2.INTER_CUBIC)
        sep_inputs.append(x)
        outputs = outputs[depth:]


    """ save output to 'nii.gz' files """
    for path, output, header, affine in zip(FLAGS.inputs, sep_inputs, headers, affines):
        dir, basename = os.path.split(path)
        seg_file_name = 'seg_' + basename
        output_save_path = os.path.join(dir, seg_file_name)

        img = nib.Nifti1Image(output, affine, header=header)
        # img.header.get_xyzt_units()
        img.to_filename(output_save_path)

########################################################################################################################
""" Pre-processing """
########################################################################################################################
def pre_processing(paths):
    def resize(input):
        stack = []
        for x in input:
            stack.append(cv2.resize(x, input_shape[:2]))
        output = np.stack(stack, 0)
        return output

    arrs = []
    sizes = []
    for path in paths:
        data = nib.load(path)
        ndarray = data.get_fdata()
        ndarray = np.transpose(ndarray, [2, 0, 1])
        ndarray = resize(ndarray)
        ndarray = np.expand_dims(ndarray, -1)
        size = ndarray.shape[0]
        arrs.append(ndarray)
        sizes.append(size)
    arrs = np.concatenate(arrs, 0)
    arrs = arrs.astype('float32')
    return arrs, sizes

########################################################################################################################