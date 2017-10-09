import tensorflow as tf
import metadata
import featurizer
import parsers
import preprocess


def json_serving_input_fn():

    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.FEATURE_NAMES]

    inputs = {}

    for column in input_feature_columns:
        inputs[column.name] = tf.placeholder(shape=[None], dtype=column.dtype)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }

    if metadata.TASK_TYPE == "custom":
        return tf.estimator.export.ServingInputReceiver(
            features=preprocess.process_features(features), receiver_tensors=inputs)

    return tf.contrib.learn.InputFnOps(
        preprocess.process_features(features),
        None,
        inputs
    )


def csv_serving_input_fn():

    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )

    features = parsers.parse_csv(csv_row)
    features.pop(metadata.TARGET_NAME)

    if metadata.TASK_TYPE == "custom":
        return tf.estimator.export.ServingInputReceiver(
            features=preprocess.process_features(features), receiver_tensors={'csv_row': csv_row}
        )

    return tf.contrib.learn.InputFnOps(
        preprocess.process_features(features),
        None,
        {'csv_row': csv_row}
    )


def example_serving_input_fn():

    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.FEATURE_NAMES]

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

    if metadata.TASK_TYPE == "custom":
        return tf.estimator.export.ServingInputReceiver(
            features=preprocess.process_features(features), receiver_tensors={'example_proto': example_bytestring})

    return tf.contrib.learn.InputFnOps(
        preprocess.process_features(features),
        None,  # labels
        {'example_proto': example_bytestring}
    )


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}