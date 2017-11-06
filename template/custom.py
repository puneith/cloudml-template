import tensorflow as tf
import parameters
import featurizer
import metadata


def regression_model_fn(features, labels, mode, params):
    """Model function for regressor with n hidden layer"""

    hidden_units = list(map(int, params.hidden_units.split(',')))
    output_layer_size = 1

    feature_columns = list(featurizer.create_feature_columns().values())
    deep_columns, _ = featurizer.get_deep_and_wide_columns(
        feature_columns,
        embedding_size=parameters.HYPER_PARAMS.embedding_size
    )

    # Create the input layers from the features
    input_layer = tf.feature_column.input_layer(features, deep_columns)

    # Create a fully-connected layer-stack based on the hidden_units in the params
    hidden_layers = tf.contrib.layers.stack(inputs= input_layer,
                                            layer= tf.contrib.layers.fully_connected,
                                            stack_args= hidden_units)

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(hidden_layers, output_layer_size)

    # Reshape output layer to 1-dim Tensor to return predictions
    output = tf.squeeze(logits)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:

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
    loss = tf.losses.mean_squared_error(labels, output)

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


def classification_model_fn(features, labels, mode, params):
    """Model function for classifier with 1 hidden layer"""

    hidden_units = list(map(int, params.hidden_units.split(',')))
    output_layer_size = len(metadata.TARGET_LABELS)

    feature_columns = list(featurizer.create_feature_columns().values())
    deep_columns, _ = featurizer.get_deep_and_wide_columns(
        feature_columns,
        embedding_size=parameters.HYPER_PARAMS.embedding_size
    )

    # Create the input layers from the features
    input_layer = tf.feature_column.input_layer(features=features,
                                                feature_columns=deep_columns)

    # Create a fully-connected layer-stack based on the hidden_units in the params
    hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
                                            layer=tf.contrib.layers.fully_connected,
                                            stack_args=hidden_units)

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=hidden_layers,
                             units=output_layer_size)

    # Reshape output layer to 1-dim Tensor to return predictions
    output = tf.squeeze(logits)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(metadata.TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # Calculate loss using softmax cross entropy
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        optimizer = tf.train.AdamOptimizer()

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(metadata.TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities)
        }

        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)


