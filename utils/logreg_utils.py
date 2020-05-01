import logging
import os
import sys
import time
from timeit import default_timer as timer
from multiprocessing import Pool, Manager, RawArray
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from utils.logging_utils import compute_sklearn_stats

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')

# A global dictionary storing the variables passed from the initializer
var_dict = {}

def init_worker(args, X_train, X_train_shape, y_train, y_train_shape, X_test, X_test_shape, y_test, y_test_shape):
    var_dict['args'] = args
    var_dict['X_train'] = X_train
    var_dict['X_train_shape'] = X_train_shape
    var_dict['y_train'] = y_train
    var_dict['y_train_shape'] = y_train_shape
    var_dict['X_test'] = X_test
    var_dict['X_test_shape'] = X_test_shape
    var_dict['y_test'] = y_test
    var_dict['y_test_shape'] = y_test_shape

def worker_func(input):
    # Get current C value and split index
    C, split = input

    # Get variables from global var_dict
    args = var_dict['args']
    X_train_np = np.frombuffer(var_dict['X_train']).reshape(var_dict['X_train_shape'])
    y_train_np = np.frombuffer(var_dict['y_train']).reshape(var_dict['y_train_shape'])
    X_test_np = np.frombuffer(var_dict['X_test']).reshape(var_dict['X_test_shape'])
    y_test_np = np.frombuffer(var_dict['y_test']).reshape(var_dict['y_test_shape'])

    # Get meta-train and meta-test features/targets
    train_features = X_train_np[split]
    train_targets = y_train_np[split]
    test_features = X_test_np[split]
    test_targets = y_test_np[split]

    if args.mp_verbose:
        logger.info('started fitting [C={}, split={}]'.format(C, split))

    # Given current C and split, fit a logistic regression classifier to the meta-train data
    timer_start = timer()
    classifier = LogisticRegression(
        multi_class=args.multi_class,
        solver=args.solver,
        n_jobs=None,
        tol=args.tol,
        C=C,
        random_state=args.seed,
        max_iter=args.max_iter,
        verbose=args.verbose
    ).fit(train_features, train_targets)
    timer_end = timer()
    if args.mp_verbose:
        logger.info('finished fitting [C={}, split={}] in {:.2f} seconds'.format(C, split, timer_end-timer_start))

    # Evaluate the trained classifier on the meta-test data, using the specified eval metric
    test_predictions = classifier.predict(test_features)
    test_probabilities = classifier.predict_proba(test_features)
    stats = compute_sklearn_stats(test_targets, test_predictions, test_probabilities, args.num_classes, args.eval_metric)

    return stats[args.eval_metric]

def LogRegCV(args, X_data, y_data):
    n_splits = args.n_splits
    n_features = X_data.shape[1]

    # Split original train dataset into meta-train and meta-test datasets
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=args.test_size, random_state=0)
    n_train_examples = len(next(iter(sss.split(X_data, y_data)))[0])
    n_test_examples = len(next(iter(sss.split(X_data, y_data)))[1])

    # Initialize arrays for storing meta-train and meta-test data
    X_train_data = np.zeros((n_splits, n_train_examples, n_features))
    y_train_data = np.zeros((n_splits, n_train_examples))
    X_test_data = np.zeros((n_splits, n_test_examples, n_features))
    y_test_data = np.zeros((n_splits, n_test_examples))

    # Get shape of meta-train and meta-test data arrays
    X_train_shape = (n_splits, n_train_examples, n_features)
    y_train_shape = (n_splits, n_train_examples)
    X_test_shape = (n_splits, n_test_examples, n_features)
    y_test_shape = (n_splits, n_test_examples)

    # Copy data from train data array into meta-train and meta-test data arrays
    for i, (train_index, test_index) in enumerate(sss.split(X_data, y_data)):
        X_train_data[i,:,:] = X_data[train_index,:]
        X_test_data[i,:,:] = X_data[test_index,:]
        y_train_data[i,:] = y_data[train_index]
        y_test_data[i,:] = y_data[test_index]

    # Initialize RawArrays for storing meta-train and meta-test data
    X_train = RawArray('d', X_train_shape[0] * X_train_shape[1] * X_train_shape[2])
    y_train = RawArray('d', y_train_shape[0] * y_train_shape[1])
    X_test = RawArray('d', X_test_shape[0] * X_test_shape[1] * X_test_shape[2])
    y_test = RawArray('d', y_test_shape[0] * y_test_shape[1])

    # Wrap RawArrays as numpy arrays so we can easily manipulate them
    X_train_np = np.frombuffer(X_train).reshape(X_train_shape)
    y_train_np = np.frombuffer(y_train).reshape(y_train_shape)
    X_test_np = np.frombuffer(X_test).reshape(X_test_shape)
    y_test_np = np.frombuffer(y_test).reshape(y_test_shape)
    
    # Copy meta-train and meta-test data to the shared numpy arrays
    np.copyto(X_train_np, X_train_data)
    np.copyto(y_train_np, y_train_data)
    np.copyto(X_test_np, X_test_data)
    np.copyto(y_test_np, y_test_data)
    
    # Start the process pool and fit a logistic regression classifier for each (C, split)
    # Here, we pass each data array and its shape to the initializer of each worker
    C_vals = np.array(args.C_vals)
    with Pool(
        processes=args.num_workers_sklearn, 
        initializer=init_worker, 
        initargs=(
            args,
            X_train, 
            X_train_shape, 
            y_train, 
            y_train_shape, 
            X_test, 
            X_test_shape, 
            y_test, 
            y_test_shape 
        )
    ) as pool:
        scores = pool.map(worker_func, product(C_vals, range(n_splits)))
        scores_arr = np.array(scores).reshape(len(C_vals), n_splits)

    # For each C value, compute mean score across all splits
    C_mean_scores = np.mean(scores_arr, axis=1)

    # Return the best C value
    best_C = C_vals[np.argmax(C_mean_scores)]
    best_avg_score = np.max(C_mean_scores)
    if args.mp_verbose:
        logger.info('best_C={}, best_avg_score={:.6f} ({})'.format(best_C, best_avg_score, args.eval_metric))

    return best_C
