import argparse
import os

import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

import parameters
import experiment
import serving


# entry point
def main():

    args_parser = argparse.ArgumentParser()
    args = parameters.initialise_arguments(args_parser)
    parameters.HYPER_PARAMS = hparam.HParams(**args.__dict__)

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[args.verbosity] / 10)

    # Directory to store output model and checkpoints
    output_dir = args.job_dir

    # If job_dir_reuse is False then remove the job_dir if it exists
    if not args.reuse_job_dir:
        if tf.gfile.Exists(args.job_dir):
            tf.gfile.DeleteRecursively(args.job_dir)
            tf.logging.info("Deleted job_dir {} to avoid re-use".format(args.job_dir))
        else:
            tf.logging.info("No job_dir available to delete")
    else:
        tf.logging.info("Reusing job_dir {} if it exists".format(args.job_dir))

    # Run the training experiment
    learn_runner.run(
        experiment_fn=experiment.generate_experiment_fn(
            min_eval_frequency=args.min_eval_frequency,
            eval_delay_secs=args.eval_delay_secs,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            export_strategies=[saved_model_export_utils.make_export_strategy(
                serving.SERVING_FUNCTIONS[args.export_format],
                exports_to_keep=1
            )]
        ),
        run_config=tf.contrib.learn.RunConfig(
            model_dir=output_dir,
            log_device_placement=True
        ),
        schedule="train_and_evaluate",
        hparams=parameters.HYPER_PARAMS
    )


if __name__ == '__main__':
    main()
