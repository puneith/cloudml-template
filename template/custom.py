import tensorflow as tf
import parameters
import featurizer


def create_estimator(config):
    return tf.estimator.Estimator(model_fn=generate_regression_model_fn, params=parameters.HYPER_PARAMS, config=config)


def generate_regression_model_fn(features, labels, mode, params):
    """Model function for Estimator with 1 hidden layer"""

    hidden_units = list(map(int, params.hidden_units.split(',')))
    hidden_layer_size = hidden_units[0]
    output_layer_size = 1

    feature_columns = list(featurizer.create_feature_columns().values())
    deep_columns, _ = featurizer.get_deep_and_wide_columns(
        feature_columns,
        embedding_size=parameters.HYPER_PARAMS.embedding_size
    )

    # Create the input layers from the features
    input_layer = tf.feature_column.input_layer(features, deep_columns)

    # Connect the input layer to the hidden layer
    hidden_layer = tf.layers.dense(input_layer, hidden_layer_size, activation=tf.nn.relu)

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    output_layer = tf.layers.dense(hidden_layer, output_layer_size)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:

        # Reshape output layer to 1-dim Tensor to return predictions
        output = tf.reshape(output_layer, [-1])

        predictions = {
            'scores': output
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, output_layer)

    # Create Optimiser
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate)

    # Create training operation
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            labels, output_layer)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                loss=loss,
                                                train_op=train_op,
                                                eval_metric_ops=eval_metric_ops)
    return estimator_spec


def generate_classification_model_fn(features, labels, mode, params):
    """Model function for Estimator with 1 hidden layer"""

    hidden_units = list(map(int, params.hidden_units.split(',')))
    hidden_layer_size = hidden_units[0]
    output_layer_size = 1

    feature_columns = list(featurizer.create_feature_columns().values())
    deep_columns, _ = featurizer.get_deep_and_wide_columns(
        feature_columns,
        embedding_size=parameters.HYPER_PARAMS.embedding_size
    )

    # Create the input layers from the features
    input_layer = tf.feature_column.input_layer(features, deep_columns)

    # Connect the input layer to the hidden layer
    hidden_layer = tf.layers.dense(input_layer, hidden_layer_size, activation=tf.nn.relu)

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    output_layer = tf.layers.dense(hidden_layer, output_layer_size)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:

        # Reshape output layer to 1-dim Tensor to return predictions
        output = tf.reshape(output_layer, [-1])

        predictions = {
            'scores': output
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, output_layer)

    # Create Optimiser
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate)

    # Create training operation
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            labels, output_layer)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                loss=loss,
                                                train_op=train_op,
                                                eval_metric_ops=eval_metric_ops)
    return estimator_spec

