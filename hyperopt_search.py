import math
import time
from pprint import pprint
import pandas as pd

import hyperopt
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from pymongo import MongoClient

from sacred_wrapper import hyperopt_objective
from hopt_sacred import hyperopt_grid, to_exp_space, objective_wrapper

import warnings
# about filterwarnings, see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


def MNIST_CNN_grid_ranges():
    """ Hyper params were chosen according to original ZS experiment"""

    main_config = dict(model_name='CNN')
    arguments = dict(debug=1)

    # Define the search space (grid search):
    #   in log space
    grid_log_ranges = dict(lr=(-2.5, -1.5, 0.5), # 3e-3, 3e-2
                           fc_dim=(1., 2.5, 0.25),  # 10, 20, 30, 60, 100, 200, 300
                           )

    #   in regular "linear" space
    grid_lin_ranges = dict(dropout_rate=(0.4, 0.7, 0.1),) # .4, .5, .6, .7,

    #   in categorical space
    grid_categorical = dict()

    return main_config, grid_log_ranges, grid_lin_ranges, grid_categorical, arguments


def main():
    mongo_port = 27017
    hyperopt_db_name = 'hyperopt_mnist'

    prepare_grid_ranges = MNIST_CNN_grid_ranges
    print('Experiment func name:', prepare_grid_ranges.__name__)
    main_config, grid_log_ranges, grid_lin_ranges, grid_categorical, arguments = \
        prepare_grid_ranges()

    # Prepare the objective to search on
    model_obj = objective_wrapper(hyperopt_objective, main_config, arguments)
    experiment_name = f'mnist_{model_obj.name}'

    # Delete old experiment-search from hyperopt db (if exists).
    # Note: This does not delete the sacred experiments, associated with it. In
    # fact, if a sacred experiment exists for some hyper-param, its result will
    # be reused.
    delete_hopt_search_from_mongo_db(experiment_name, hyperopt_db_name, mongo_port)

    # Define the grid to search upon
    grid = hyperopt_grid(grid_log_ranges, grid_lin_ranges, grid_categorical)

    # pretty print the grid search ranges
    pprint_experiments(main_config, grid, arguments)
    count_down(20)

    trials = MongoTrials(f'mongo://localhost:{mongo_port}/{hyperopt_db_name}/jobs',
                         exp_key=experiment_name)
    print('Executing random order grid search. Pending on workers to connect ..')
    argmin = fmin(fn=model_obj.objective,
                 space=grid.space,
                algo=grid.suggest,
                # max_evals=grid.num_combinations,
                max_evals=30,
                trials=trials,
                verbose=1)

    process_and_print_results(trials, grid)





def process_and_print_results(trials, grid):
    """ Aggregate results to a pandas DataFrame and display them. """
    df_results = pd.DataFrame([dict(t['misc']['vals'], **dict(t['result'])) for t in trials.trials])
    df_results = df_results.applymap(extract_first_if_list)
    df_results['minimize_metric'] = df_results['loss']
    df_results = df_results.drop(['loss'], axis=1)

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print('all experiments:\n', df_results)

    print('\n\nGrid ranges were:')
    grid.pprint()

    print('\nBest:\n', df_results.iloc[df_results.minimize_metric.idxmin, :])

def delete_hopt_search_from_mongo_db(experiment_name, hyperopt_db_name, mongo_port):
    """Deletes old experiment-search from hyperopt db (if exists).
    Note: This does not delete the sacred experiments, associated with it. In
    fact, if a sacred experiment exists for some hyper-param, its result will
    be reused.
    """
    client = MongoClient('localhost', mongo_port)
    db_hopt = client[hyperopt_db_name]
    result = db_hopt.jobs.delete_many({'exp_key': {'$regex': experiment_name}})
    print(f'Deleted {result.deleted_count} record(s)')

def count_down(T):
    """
    :type T: int
    """
    print('\nWaiting for %d seconds. press ctrl-C to break' % T)
    for t in range(T):
        if T-t<=5:
            print(f'{T-t}', end='', flush=True)
        time.sleep(1)
    print('')

def pprint_experiments(main_config, grid, arguments):
    print('\n\nMain configuration is')
    pprint(main_config)
    print('\nGrid search over:')
    grid.pprint()
    print('grid.num_combinations = ', grid.num_combinations)
    print('\nAdditional commandline arguments:')
    pprint(arguments)
    if arguments.get('debug', 0):
        print('\x1b[0;30;41m' + 'NOTE: DEBUG MODE IS SET!' + '\x1b[0m')
    else:
        print('\x1b[6;30;42m' + 'NOTE: DEBUG MODE IS OFF!' + '\x1b[0m')

def extract_first_if_list(x):
    if isinstance(x, list):
        return x[0]
    return x


if __name__ == '__main__':
    main()