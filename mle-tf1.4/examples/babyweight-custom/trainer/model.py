# ************************************************************************************
# YOU MAY MODIFY THIS MODULE TO USE DIFFERENT ESTIMATORS OR CONFIGURE THE CURRENT ONES
# ************************************************************************************
# YOU NEED TO MODIFY THIS MODULE IF YOU WANT TO IMPLEMENT A CUSTOM ESTIMATOR
# ************************************************************************************

import tensorflow as tf

import featurizer
import parameters
import metadata


def create_classifier(config):
    """ Create a DNNLinearCombinedClassifier based on the HYPER_PARAMS in the parameters module

    Args:
        config - used for model directory
    Returns:
        DNNLinearCombinedClassifier
    """

    feature_columns = list(featurizer.create_feature_columns().values())

    deep_columns, wide_columns = featurizer.get_deep_and_wide_columns(
        feature_columns
    )

    linear_optimizer = tf.train.FtrlOptimizer(learning_rate=parameters.HYPER_PARAMS.learning_rate)
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=parameters.HYPER_PARAMS.learning_rate)

    classifier = tf.estimator.DNNLinearCombinedClassifier(

        n_classes=len(metadata.TARGET_LABELS),
        label_vocabulary=metadata.TARGET_LABELS,

        linear_optimizer=linear_optimizer,
        linear_feature_columns=wide_columns,

        dnn_feature_columns=deep_columns,
        dnn_optimizer=dnn_optimizer,

        weight_column=metadata.WEIGHT_COLUMN_NAME,

        dnn_hidden_units=construct_hidden_units(),
        dnn_activation_fn=tf.nn.relu,
        dnn_dropout=parameters.HYPER_PARAMS.dropout_prob,

        config=config,
    )

    print("creating a classification model: {}".format(classifier))

    return classifier


def create_regressor(config):
    """ Create a DNNLinearCombinedRegressor based on the HYPER_PARAMS in the parameters module

    Args:
        config - used for model directory
    Returns:
        DNNLinearCombinedRegressor
    """

    feature_columns = list(featurizer.create_feature_columns().values())

    deep_columns, wide_columns = featurizer.get_deep_and_wide_columns(
        feature_columns
    )

    linear_optimizer = tf.train.FtrlOptimizer(learning_rate=parameters.HYPER_PARAMS.learning_rate)
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=parameters.HYPER_PARAMS.learning_rate)

    regressor = tf.estimator.DNNLinearCombinedRegressor(

        linear_optimizer=linear_optimizer,
        linear_feature_columns=wide_columns,

        dnn_feature_columns=deep_columns,
        dnn_optimizer=dnn_optimizer,

        weight_column=metadata.WEIGHT_COLUMN_NAME,

        dnn_hidden_units=construct_hidden_units(),
        dnn_activation_fn=tf.nn.relu,
        dnn_dropout=parameters.HYPER_PARAMS.dropout_prob,

        config=config,
    )

    print("creating a regression model: {}".format(regressor))

    return regressor


def create_estimator(config):
    """ Create a custom estimator based on _model_fn

    Args:
        config - used for model directory
    Returns:
        Estimator
    """

    def _model_fn(features, labels, mode, params):
        """ model function for the custom estimator"""

        hidden_units = construct_hidden_units()
        output_layer_size = 1  # because it is a regression problem

        feature_columns = list(featurizer.create_feature_columns().values())

        # create the deep columns: dense + indicators
        deep_columns, _ = featurizer.get_deep_and_wide_columns(
            feature_columns
        )

        # Create input layer based on features
        input_layer = tf.feature_column.input_layer(features=features,
                                                    feature_columns=deep_columns)

        # Create hidden layers (cnn, rnn, dropouts, etc.) given the input layer
        hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
                                                layer=tf.contrib.layers.fully_connected,
                                                stack_args=hidden_units,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

        # Create output (logits) layer given the hidden layers
        logits = tf.layers.dense(inputs=hidden_layers,
                                 units=output_layer_size,
                                 activation=None)

        # Reshape output layer to 1-dim Tensor to return predictions
        output = tf.squeeze(logits)

        # Specify the model output (i.e. predictions) given the output layer
        predictions = {
            'scores': output
        }

        # Specify the export output based on the predictions
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        loss = None
        train_op = None
        eval_metric_ops = None

        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            # Calculate loss using mean squared error
            loss = tf.losses.mean_squared_error(labels, output)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create Optimiser
            optimizer = tf.train.AdamOptimizer(
                learning_rate=params.learning_rate)

            # Create training operation
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.EVAL:
            # Specify root mean squared error as additional eval metric
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(labels, output)
            }

        # Provide an estimator spec
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                    loss=loss,
                                                    train_op=train_op,
                                                    eval_metric_ops=eval_metric_ops,
                                                    predictions=predictions,
                                                    export_outputs=export_outputs
                                                    )
        return estimator_spec

    print("creating a custom model...")

    return tf.estimator.Estimator(model_fn=_model_fn,
                                  params=parameters.HYPER_PARAMS,
                                  config=config)


def construct_hidden_units():
    """ Create the number of hidden units in each layer

    if the HYPER_PARAMS.layer_sizes_scale_factor > 0 then it will use a "decay" mechanism
    to define the number of units in each layer. Otherwise, parameters.HYPER_PARAMS.hidden_units
    will be used as-is.

    Returns:
        list of int
    """
    hidden_units = list(map(int, parameters.HYPER_PARAMS.hidden_units.split(',')))

    if parameters.HYPER_PARAMS.layer_sizes_scale_factor > 0:
        first_layer_size = hidden_units[0]
        scale_factor = parameters.HYPER_PARAMS.layer_sizes_scale_factor
        num_layers = parameters.HYPER_PARAMS.num_layers

        hidden_units = [
            max(2, int(first_layer_size * scale_factor ** i))
            for i in range(num_layers)
        ]

    print("Hidden units structure: {}".format(hidden_units))

    return hidden_units
