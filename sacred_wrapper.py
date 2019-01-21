import sys

import mnist_keras

import warnings
# see https://stackoverflow.com/a/40846742
from hopt_sacred import hyperopt_sacred_wrapper

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

common_args = mnist_keras.args

# Inherits hyperopt_sacred_wrapper, and implements the naming methods
class mnist_experiment(hyperopt_sacred_wrapper):
    def __init__(self, f_main, f_config, f_capture, cfg,
                 mongo_url='127.0.0.1:27017', disable_logging=False):

        self.is_debug = cfg['debug']
        super(mnist_experiment, self).__init__(f_main=f_main, f_config=f_config,
                                               f_capture=f_capture, cfg=cfg,
                                               mongo_url=mongo_url,
                                               disable_logging=disable_logging)

    def sacred_db_name(self):
        str_debug = 'debug' if self.is_debug == 1 else ''
        return 'MNIST' + str_debug


    def sacred_ex_name(self):
        return 'MNIST_CNN'

# noinspection PyUnusedLocal
def ex_config():
    """ Set SACRED configuration and the default values (it will override argparse
        defaults). Note that we take only the hyper-params. Other arguments like
        --verbose, or --gpu_memory_fraction are irrelevant for the model.
    """
    seed = 0
    lr = 1e-3
    dropout_rate = 0.5
    fc_dim = 128
    epochs = 20
    batch_size = 32
    model_name = 'CNN'

# implementing the "f_main" API
def main(ex, _run, f_log_metrics):
    """ Updates the main experiment function arguments, calls it and save the
        experiment results and artifacts.
    """

    # Override argparse arguments with sacred arguments
    command_line_args = mnist_keras.args  # argparse command line arguments
    vars(command_line_args).update(_run.config)

    # call main script
    val_acc, test_acc = mnist_keras.main(f_log_metrics=f_log_metrics)

    # save the result metrics to db
    _run.info['model_metrics'] = dict(val_acc=val_acc, test_acc=test_acc)
    # save an artifact (CNN model) to db
    ex.add_artifact('mnist_model.h5')

    return val_acc


def log_metrics(_run, logs):
    """
    Implements the metrics logging API
    """
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("acc", float(logs.get('acc')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_acc", float(logs.get('val_acc')))
    _run.result = float(logs.get('val_acc'))



##### HyperOpt support #####################
from hyperopt import STATUS_OK
# noinspection PyUnresolvedReferences
import hopt_sacred
def hyperopt_objective(config, arguments):

    # get the integer arguments
    common_args_ints = [k for k, v in vars(common_args).items() if isinstance(v, int)]

    # cast to int relevant params
    for (key, value) in config.items():
        if key in common_args_ints:
            value = int(value)
        config[key] = value

    # Override argparse arguments with search arguments
    vars(common_args).update(arguments)

    experiment = mnist_experiment(f_main=main, f_config=ex_config,
                                               f_capture=log_metrics,
                                               cfg=dict(vars(common_args), **config))

    # Run experiment with given configuration, and handle duplicate experiments
    try:
        ex_name = experiment.sacred_ex_name()
        run = experiment.ex.run(config_updates=config, options={'--name': ex_name})
        ex_res = vars(run)
    except hopt_sacred.DuplicateExperiment as e:
        ex_res = e.ex

    # Hyperopt API requires a metric to minimize named as 'loss'
    err_rate = ex_res['result']
    metrics = {'loss': 1 - err_rate, 'status': STATUS_OK}

    # add '*' prefix to indicate measured model metrics
    for k, v in ex_res['info']['model_metrics'].items():
        metrics['*' + k] = v

    return metrics

##### End HyperOpt support #################



if __name__ == '__main__':
    """ This part is for calling the script directly from commandline instead of 
    calling it from hyperopt search
    """

    argv = sys.argv
    # Remove the argparse arguments, since they have already been processed by
    # argparse library, and they interfere with the SACRED command line parser
    sacred_argv = [arg for arg in argv if not arg.startswith('--')]

    experiment = mnist_experiment(f_main=main, f_config=ex_config,
                                               f_capture=log_metrics,
                                               cfg=vars(common_args))
    experiment.ex.run_commandline(sacred_argv)
