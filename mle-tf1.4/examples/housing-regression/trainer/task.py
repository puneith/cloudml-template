# ****************************************************************************
# YOU NEED NOT TO CHANGE THIS MODULE
# ****************************************************************************

import os
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import metadata
import input
import model
import parameters
import serving


def run_experiment(run_config):
    """Run the training and evaluate using the high level API"""

    train_input_fn = lambda: input.dataset_input_fn(
        file_names_pattern=parameters.HYPER_PARAMS.train_files,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_epochs=parameters.HYPER_PARAMS.num_epochs,
        batch_size=parameters.HYPER_PARAMS.train_batch_size
    )

    eval_input_fn = lambda: input.dataset_input_fn(
        file_names_pattern=parameters.HYPER_PARAMS.eval_files,
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=parameters.HYPER_PARAMS.eval_batch_size
    )

    exporter = tf.estimator.FinalExporter(
        'estimator',
        serving.SERVING_FUNCTIONS[parameters.HYPER_PARAMS.export_format],
        as_text=False  # change to true if you want to export the model as readable text
    )

    # compute the number of training steps based on num_epoch, train_size, and train_batch_size
    if parameters.HYPER_PARAMS.train_size is not None and parameters.HYPER_PARAMS.num_epochs is not None:
        train_steps = (parameters.HYPER_PARAMS.train_size / parameters.HYPER_PARAMS.train_batch_size) * \
                      parameters.HYPER_PARAMS.num_epochs
    else:
        train_steps = parameters.HYPER_PARAMS.train_steps

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=int(train_steps)
    )

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=parameters.HYPER_PARAMS.eval_steps,
        exporters=[exporter],
        name='estimator-eval',
        throttle_secs=parameters.HYPER_PARAMS.eval_every_secs,
    )

    print("* experiment configurations")
    print("==========================")
    print("Train size: {}".format(parameters.HYPER_PARAMS.train_size))
    print("Epoch count: {}".format(parameters.HYPER_PARAMS.num_epochs))
    print("Train batch size: {}".format(parameters.HYPER_PARAMS.train_batch_size))
    print("Training steps: {} ({})".format(int(train_steps),
                                           "supplied" if parameters.HYPER_PARAMS.train_size is None else "computed"))
    print("Evaluate every {} seconds".format(parameters.HYPER_PARAMS.eval_every_secs))
    print("==========================")

    if metadata.TASK_TYPE == "classification":
        estimator = model.create_classifier(
            config=run_config
        )
    elif metadata.TASK_TYPE == "regression":
        estimator = model.create_regressor(
            config=run_config
        )
    else:
        estimator = None

    # train and evaluate
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


# entry point
def main():
    args_parser = argparse.ArgumentParser()
    args = parameters.initialise_arguments(args_parser)
    parameters.HYPER_PARAMS = hparam.HParams(**args.__dict__)

    print('')
    print('Hyper-parameters:')
    print(parameters.HYPER_PARAMS)
    print('')

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[args.verbosity] / 10)

    # Directory to store output model and checkpoints
    model_dir = args.job_dir

    # If job_dir_reuse is False then remove the job_dir if it exists
    print("Resume training:", args.reuse_job_dir)
    if not args.reuse_job_dir:
        if tf.gfile.Exists(args.job_dir):
            tf.gfile.DeleteRecursively(args.job_dir)
            print("Deleted job_dir {} to avoid re-use".format(args.job_dir))
        else:
            print("No job_dir available to delete")
    else:
        print("Reusing job_dir {} if it exists".format(args.job_dir))

    run_config = tf.estimator.RunConfig(
        tf_random_seed=19830610,
        log_step_count_steps=1000,
        save_checkpoints_secs=60,
        keep_checkpoint_max=3,
        model_dir=model_dir
    )

    run_config = run_config.replace(model_dir=model_dir)
    print("Model Directory:", run_config.model_dir)

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    print("")
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")

    run_experiment(run_config)

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print("")


if __name__ == '__main__':
    main()
