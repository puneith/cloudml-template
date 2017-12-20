# ************************************************************************************
# YOU MAY MODIFY THIS MODULE TO USE DIFFERENT ESTIMATORS OR CONFIGURE THE CURRENT ONES
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

    print("training a classifier: {}".format(classifier))

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

    print("training a regressor: {}".format(regressor))

    return regressor


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
