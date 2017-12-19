# ****************************************************************************
# YOU NEED NOT TO CHANGE THIS MODULE
# ****************************************************************************


import tensorflow as tf

import metadata
import input
import featurizer


def json_serving_input_fn():
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    inputs = {}

    for column in input_feature_columns:
        if column.name in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY:
            inputs[column.name] = tf.placeholder(shape=[None], dtype=tf.int32)
        else:
            inputs[column.name] = tf.placeholder(shape=[None], dtype=column.dtype)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features=input.process_features(features),
        receiver_tensors=inputs
    )


def csv_serving_input_fn():
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )

    features = input.parse_csv_row(csv_row, is_serving=True)

    unused_features = list(set(metadata.SERVING_COLUMNS) - set(metadata.INPUT_FEATURE_NAMES) - {metadata.TARGET_NAME})

    # Remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)

    return tf.estimator.export.ServingInputReceiver(
        features=input.process_features(features),
        receiver_tensors={'csv_row': csv_row}
    )


def example_serving_input_fn():
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    feature_scalars = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(input_feature_columns)
    )

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_scalars.iteritems()
    }

    return tf.estimator.export.ServingInputReceiver(
        features=input.process_features(features),
        receiver_tensors={'example_proto': example_bytestring}
    )


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}
