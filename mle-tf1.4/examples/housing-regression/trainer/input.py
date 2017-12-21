# **************************************************************************
# YOU MAY IMPLEMENT process_features FUCNTION FOR CUSTOM FEATURE ENGINEERING
# **************************************************************************

import json
import os
import multiprocessing

import tensorflow as tf
from tensorflow import data

import metadata
import parameters


def parse_csv(csv_row, is_serving=False):
    """Takes the string input tensor (csv) and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
    (inferred from the metadata)

    Args:
        csv_row: rank-2 tensor of type string (csv)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_row input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensor of the correct data type
    """

    if is_serving:
        column_names = metadata.SERVING_COLUMNS
        defaults = metadata.SERVING_DEFAULTS
    else:
        column_names = metadata.HEADER
        defaults = metadata.HEADER_DEFAULTS

    columns = tf.decode_csv(tf.expand_dims(csv_row, -1), record_defaults=defaults)
    features = dict(zip(column_names, columns))

    return features


def parse_tf_example(example_proto, is_serving=False):
    """Takes the string input tensor (example proto) and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
    (inferred from the  metadata)

    Args:
        example_proto: rank-2 tensor of type string (example proto)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_row input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensor of the correct data type
    """

    feature_spec = {}

    for feature_name in metadata.NUMERIC_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.float32)

    for feature_name in metadata.CATEGORICAL_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.string)

    if not is_serving:

        if metadata.TASK_TYPE == 'regression':
            feature_spec[metadata.TARGET_NAME] = tf.FixedLenFeature(shape=1, dtype=tf.float32)
        else:
            feature_spec[metadata.TARGET_NAME] = tf.FixedLenFeature(shape=(), dtype=tf.string)

    parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)

    return parsed_features


def process_features(features):
    """ Use to implement custom feature engineering logic, e.g. polynomial expansion
    Default behaviour is to return the original feature tensors dictionary as is

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
        {string:tensors}: extended feature tensors dictionary
    """

    features['CRIM'] = tf.log(features['CRIM'] + 0.01)
    features['B'] = tf.clip_by_value(features['B'], clip_value_min=300, clip_value_max=500)

    return features


def dataset_input_fn(file_names_pattern,
                     file_encoding='csv',
                     mode=tf.estimator.ModeKeys.EVAL,
                     skip_header_lines=0,
                     num_epochs=1,
                     batch_size=200,
                     multi_threading=True):
    """An input function for training or evaluation.
    This uses the Dataset APIs.

    Args:
        file_names_pattern: [str] - file name or file name patterns from which to read the data.
        mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of data.
        file_encoding: type of the text files. Can be 'csv' or 'tfrecords'
        skip_header_lines: int set to non-zero in order to skip header lines
          in CSV files.
        num_epochs: int - how many times through to read the data.
          If None will loop through data indefinitely
        batch_size: int - first dimension size of the Tensors returned by
          input_fn
        multi_threading: boolean - indicator to use multi-threading or not
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """

    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

    data_size = parameters.HYPER_PARAMS.train_size if mode == tf.estimator.ModeKeys.TRAIN else None

    num_threads = multiprocessing.cpu_count() if multi_threading else 1

    buffer_size = 2 * batch_size + 1

    print("")
    print("* data input_fn:")
    print("================")
    print("Mode: {}".format(mode))
    print("Input file(s): {}".format(file_names_pattern))
    print("Files encoding: {}".format(file_encoding))
    print("Data size: {}".format(data_size))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    file_names = tf.matching_files(file_names_pattern)

    if file_encoding == 'csv':
        dataset = data.TextLineDataset(filenames=file_names)
        dataset = dataset.skip(skip_header_lines)
        dataset = dataset.map(lambda csv_row: parse_csv(csv_row))

    else:
        dataset = data.TFRecordDataset(filenames=file_names)
        dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example),
                              num_parallel_calls=num_threads)

    dataset = dataset.map(lambda features: get_features_target_tuple(features),
                          num_parallel_calls=num_threads)
    dataset = dataset.map(lambda features, target: (process_features(features), target),
                          num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size)
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()

    return features, target


def get_features_target_tuple(features):
    unused_features = list(set(metadata.HEADER) -
                           set(metadata.INPUT_FEATURE_NAMES) -
                           {metadata.TARGET_NAME} -
                           {metadata.WEIGHT_COLUMN_NAME})

    # Remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)

    # Get target feature
    target = features.pop(metadata.TARGET_NAME)

    return features, target


def load_feature_stats():
    """
    load numeric column pre-computed statistics (mean, stdv, min, max, etc.)
    in order to be used for scaling/stretching numeric columns

    In practice, the statistics of large dataset are computed proior to model training,
    using dataflow (beam), dataproc (spark), BigQuery, etc.

    The stats are then saved to gcs location. The location is passed to package are an arg.
    Then in this function, the file is downloaded - via gsutil -  and loaded to be used.

    The following code assumes that the stats.json file is already available locally.

    Returns:
        A dict{string: float}, where key is the name of the stat
    """

    feature_stats = None
    try:
        with open("../data/stats.json") as file:
            content = file.read()
        feature_stats = json.loads(content)
        print("INFO:feature stats were successfully loaded from local file...")
    except:
        print("WARN:couldn't load feature stats. numerical columns will not be normalised...")

    return feature_stats
