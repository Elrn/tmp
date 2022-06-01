import tensorflow as tf
import utils
from os.path import basename
import numpy as np

num_class = 2
input_shape = [128, 128, 1] # Z: 200
########################################################################################################################
def parse_fn(data, seg): # RANK: (4, 3)
    # adc = tf.transpose(adc, [2, 0, 1])
    # dwi = tf.transpose(dwi, [2, 0, 1])
    # seg = tf.transpose(seg, [2, 0, 1])

    data = tf.expand_dims(data, -1)
    data = tf.image.per_image_standardization(data)
    data = tf.cast(data, 'float32')
    seg = tf.cast(seg, 'int32')
    seg = tf.one_hot(seg, num_class, axis=-1)

    return (data, seg)

def validation_split_fn(dataset, validation_split):
    len_dataset = tf.data.experimental.cardinality(dataset).numpy()
    valid_count = int(len_dataset * validation_split)
    print(f'[Dataset|split] Total: "{len_dataset}", Train: "{len_dataset-valid_count}", Valid: "{valid_count}"')
    return dataset.skip(valid_count), dataset.take(valid_count)

def build(batch_size, validation_split=0.1):
    assert 0 <= validation_split <= 0.5
    file_path = 'C:\\dataset\\dwi_image_preprocessed.npz'
    print(f'[Dataset] load:"{basename(file_path)}", batch size:"{batch_size}", split:"{validation_split}"')
    with np.load(file_path) as data:
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


def build_test(batch_size):
    file_path = 'C:\\dataset\\dwi_image_preprocessed.npz'
    with np.load(file_path) as data:
        adc = tf.concat([data['adc_lesion_x'], data['adc_nolesion_x']], -1)
        dwi = tf.concat([data['dwi_lesion_x'], data['dwi_nolesion_x']], -1)
        d = tf.concat([adc, dwi], -1)
        seg = tf.concat([data['dwi_lesion_y'], data['dwi_nolesion_y']], -1)
        seg = tf.concat([seg, seg], -1)
        d = tf.transpose(d, [2, 0, 1])
        seg = tf.transpose(seg, [2, 0, 1])
        dataset = load((d, seg), batch_size)
    return dataset

def load_test(data, batch_size, drop=True):
    return tf.data.Dataset.from_tensor_slices(data
        ).map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).unbatch(  # batch > unbatch > batch 시 cardinality = -2 로 설정됨
        ).batch(batch_size=batch_size, drop_remainder=drop,
        )
