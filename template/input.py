import tensorflow as tf
import tensorflow.contrib.data as data
import metadata
import multiprocessing


def process_features(features):
    """ Use to implement custom feature engineering logic, e.g. polynomial expansion
    Default behaviour is to return the original feature tensors dictionary as is

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
        {string:tensors}: extended feature tensor dictionary
    """

    # examples - given:
    # 'x' and 'y' are two numeric features:
    # 'alpha' and 'beta' are two categorical features

    # features['x_2'] = tf.pow(features['x'],2)
    # features['y_2'] = tf.pow(features['y'], 2)
    # features['xy'] = features['x'] * features['y']
    # features['sin_x'] = tf.sin(features['x'])
    # features['cos_y'] = tf.cos(features['x'])
    # features['log_xy'] = tf.log(features['xy'])
    # features['sqrt_xy'] = tf.sqrt(features['xy'])
    # features['x_grt_y'] = features['x'] > features['y']

    return features


def parse_csv_row(csv_row, is_serving=False):
    """Takes the string input tensor and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
    (inferred from the HEADER_DEFAULTS metadata)

    Args:
        csv_row: rank-2 tensor of type string (csv)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_string input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensor of the correct data type
    """
    header = metadata.HEADER.copy()
    defaults = metadata.HEADER_DEFAULTS.copy()

    if is_serving:
        csv_row = tf.expand_dims(csv_row, -1)
        target_index = metadata.HEADER.index(metadata.TARGET_NAME)
        defaults.pop(target_index)
        header.pop(target_index)

    columns = tf.decode_csv(csv_row, record_defaults=defaults)
    features = dict(zip(header, columns))

    return features


def parse_tf_example(example_proto, is_serving=False):
    """Takes the string input tensor (example proto) and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
    (inferred from the HEADER_DEFAULTS metadata)

    Args:
        example_proto: rank-2 tensor of type string (example proto)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving example_proto input is different than the training input (i.e., no target feature)
    Returns:
        rank-2 tensor of the correct data type
    """
    features_spec = {}

    for feature_name in metadata.NUMERIC_FEATURE_NAMES:
        features_spec[feature_name] = tf.FixedLenFeature(shape=(), dtype=tf.float32)

    for feature_name in metadata.CATEGORICAL_FEATURE_NAMES:
        features_spec[feature_name] = tf.FixedLenFeature(shape=(), dtype=tf.string)

    if is_serving:
        features_spec[metadata.TARGET_NAME] = tf.FixedLenFeature(shape=(), dtype=tf.string)

    features = tf.parse_example(serialized=example_proto, features=features_spec)

    target = features.pop(metadata.TARGET_NAME)

    return features, target


def text_input_fn(file_names,
                  parser_fn=parse_csv_row,
                  mode=tf.contrib.learn.ModeKeys.TRAIN,
                  skip_header_lines=0,
                  num_epochs=None,
                  batch_size=200):

    """An input function for training or evaluation.
    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.

    Args:
        file_names: [str] - list of text files to read data from.
        mode: tf.contrib.learn.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of data.
        parser_fn: A function that parses text files (e.g., csv parser, fixed-width parser, etc.
        skip_header_lines: int set to non-zero in order to skip header lines
          in CSV files.
        num_epochs: int - how many times through to read the data.
          If None will loop through data indefinitely
        batch_size: int - first dimension size of the Tensors returned by
          input_fn
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """

    shuffle = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False

    input_file_names = tf.train.match_filenames_once(file_names)

    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=shuffle)

    reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

    _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

    features = parser_fn(rows)

    if shuffle:
        features = tf.train.shuffle_batch(
            features,
            batch_size,
            min_after_dequeue=2 * batch_size + 1,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )
    else:
        features = tf.train.batch(
            features,
            batch_size,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )

    features, target =  get_features_target_tuple(features)

    return process_features(features), target


def dataset_input_fn(file_names,
                     parser_fn=parse_csv_row,
                     mode=tf.contrib.learn.ModeKeys.TRAIN,
                     skip_header_lines=0,
                     num_epochs=None,
                     batch_size=200):

    """An input function for training or evaluation.
    This uses the Dataset APIs.

    Args:
        file_names: [str] - list of text files to read data from.
        mode: tf.contrib.learn.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of data.
        parser_fn: A function that parses text files (e.g., csv parser, fixed-width parser, tf example parser, etc.
        skip_header_lines: int set to non-zero in order to skip header lines
          in CSV files.
        num_epochs: int - how many times through to read the data.
          If None will loop through data indefinitely
        batch_size: int - first dimension size of the Tensors returned by
          input_fn
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """

    shuffle = True if mode == tf.contrib.learn.ModeKeys.TRAIN else False

    dataset = data.TextLineDataset(filenames=file_names)
    dataset = dataset.skip(skip_header_lines)
    dataset = dataset.map(lambda csv_row: parser_fn(csv_row))
    dataset = dataset.map(lambda features: get_features_target_tuple(features))
    dataset = dataset.map(lambda features, target: (process_features(features), target))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()

    features, target = iterator.get_next()
    return features, target


def get_features_target_tuple(features):

    # Remove unused columns
    for column in metadata.UNUSED_FEATURE_NAMES:
        features.pop(column)

    # Get target feature
    target = features.pop(metadata.TARGET_NAME)

    return features, target