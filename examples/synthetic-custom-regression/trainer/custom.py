import tensorflow as tf
import parameters
import featurizer
import metadata


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
    logits = tf.layers.dense(hidden_layer, output_layer_size)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Reshape output layer to 1-dim Tensor to return predictions
        output = tf.reshape(logits, [-1])

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
    loss = tf.losses.mean_squared_error(labels, logits)

    # Create Optimiser
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params.learning_rate)

    # Create training operation
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            labels, logits)
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
    logits = tf.layers.dense(hidden_layer, output_layer_size)

    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'classes': tf.gather(metadata.TARGET_LABELS, predicted_indices),
            'scores': tf.reduce_max(probabilities, axis=1)
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                    loss=loss,
                                                    train_op=train_op)

        return estimator_spec

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=metadata.TARGET_LABELS.shape[0],
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

