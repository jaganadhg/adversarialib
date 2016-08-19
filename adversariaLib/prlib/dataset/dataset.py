#-------------------------------------------------------------------------------
# adversariaLib - Advanced library for the evaluation of machine 
# learning algorithms and classifiers against adversarial attacks.
# 
# Copyright (C) 2013, Igino Corona, Battista Biggio, Davide Maiorca, 
# Dept. of Electrical and Electronic Engineering, University of Cagliari, Italy.
# 
# adversariaLib is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# adversariaLib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------
import numpy as np
import random
import os
from util import log, get_data, save_data, get_fname_dataset, get_fname_exp
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file

def save_dataset(X, y, fname, delimiter=' '):
    f = open(fname, "w")
    for i, pattern in enumerate(X):
        f.write(delimiter.join([str(elm) for elm in pattern]))
        f.write(" %.4f\n" % y[i])
    f.close()

def save_indexes(indexes, fname):
    f = open(fname, "w")
    for index in indexes:
        f.write("%d\n" % index)
    f.close()

def get_indexes(fname):
    indexes = []
    f = open(fname, "r")
    line = f.readline()
    while line:
        indexes.append(int(line))
        line = f.readline()
    f.close()  
    return tuple(indexes)

def save_lines(lines, fname):
    f = open(fname, "w")
    for line in lines:
        f.write("%s\n" % line)
    f.close()
    
def get_scores(fname):
    scores = []
    f = open(fname, "r")
    line = f.readline()
    while line:
        scores.append(float(line))
        line = f.readline()
    f.close()  
    return tuple(scores)

def get_xy(dset, indexes):
    if indexes:
        X = np.ndarray((len(indexes), len(dset[0])-1))
        y = np.arange(len(indexes))
        for i, index in enumerate(indexes):
            X[i] = dset[index][:-1]
            y[i] = dset[index][-1]
    else:
        X = np.ndarray((len(dset), len(dset[0])-1))
        y = np.arange(len(dset))
        for i, row in enumerate(dset):
            X[i] = row[:-1]
            y[i] = row[-1]
    return X, y

def load_sparse_dataset(fname, indexes=None, **kwargs):
    """Loads a SPARSE dataset (svmlight format).
    Only one output per pattern is supported."""
    dset = load_svmlight_file(fname, **kwargs)
    if indexes is None:
        X = dset[0]
        y = dset[1]
    else:
        indexes = np.array(indexes)
        X = dset[0][indexes,:]
        y = dset[1][indexes]
    return X, y

def load_dataset(fname, indexes=None, **kwargs):
    """Loads a dataset.
    We first try to load the dataset assuming a svmlight format (SPARSE dataset).
    If not successful, we assume one pattern per row, with space-separated feature values.
    The last value in a row is considered as the target output."""
    try:
        return load_sparse_dataset(fname, indexes=indexes, **kwargs)
    except:
        dset = np.ndfromtxt(fname, delimiter=' ') # TODO: to save memory we must implement this
        return get_xy(dset, indexes=indexes)

def get_dataset_size(fname):
    """It works only if there are no empty lines (each line actually refer to one pattern)"""
    f = open(fname)
    size = f.read().count(os.linesep)
    f.close()
    return size

def check_dataset(folder, dset_name):
    dset_fname, exists = get_fname_dataset(folder, dset_name)
    if not exists:
        log("Dataset %s (%s) NOT found!" % (dset_name, dset_fname))
        exit(1)
    return dset_fname

def get_index_data(folder, dset_name, test_fraction):
    dset_fname = check_dataset(folder, dset_name)
    data = {}
    data['dset_size'] = get_dataset_size(dset_fname)
    data['ts_size'] = int(data['dset_size']*test_fraction)
    data['tr_size'] = data['dset_size']-data['ts_size']
    return data

def build_train_test_splits(exp_name, dset_folder, dset_name, test_fraction, nsplits):
    data = get_index_data(dset_folder, dset_name, test_fraction)
    indexes = set(range(data['dset_size']))
    for split_no in range(nsplits):
        dset_tr_split_fname, exists_tr = get_fname_exp(exp_name, dset_name, indexes='split', 
                                                       split_type='tr', split_no=split_no)
        dset_ts_split_fname, exists_ts = get_fname_exp(exp_name, dset_name, indexes='split', 
                                                       split_type='ts', split_no=split_no)
        if exists_tr and exists_ts:
            log("It seems that train/test splits for dataset %s have been already built." % dset_name)
            break
        elif exists_tr or exists_ts:
            log("MESS: Check train/test split %d for dataset %s." % (split_no, dset_name))
            exit(1)
        log('Creating train/test split %d for dataset %s (test fraction: %d percent)...' % (split_no, dset_name, test_fraction*100))
        ts_indexes = set(random.sample(indexes, data['ts_size']))
        assert(len(ts_indexes) == data['ts_size']) 
        tr_indexes = indexes - ts_indexes
        save_indexes(tr_indexes, dset_tr_split_fname)
        save_indexes(ts_indexes, dset_ts_split_fname)
    return data

def get_training_data(exp_name, dset_folder, dset_name, test_fraction, split_no):
    dset_fname = check_dataset(dset_folder, dset_name)
    dset_tr_split_fname, exists = get_fname_exp(exp_name, dset_name, indexes='split', 
                                                split_type='tr', split_no=split_no)
    if not exists:
        log("Dataset %s Training Split %d NOT found!" % (dset_name, split_no))
        exit(1)
    tr_indexes = get_indexes(dset_tr_split_fname)
    data = get_index_data(dset_folder, dset_name, test_fraction)
    assert(int(test_fraction*(data['tr_size']+data['ts_size'])) == data['ts_size'])
    assert(data['tr_size'] == len(tr_indexes))
    return load_dataset(dset_fname, tr_indexes)

def get_testing_data(exp_name, dset_folder, dset_name, test_fraction, split_no):
    dset_fname = check_dataset(dset_folder, dset_name)
    dset_ts_split_fname, exists = get_fname_exp(exp_name, dset_name, indexes='split', 
                                                split_type='ts', split_no=split_no)
    if not exists:
        log("Dataset %s Testing Split %d NOT found!" % (dset_name, split_no))
        exit(1)
    ts_indexes = get_indexes(dset_ts_split_fname)
    data = get_index_data(dset_folder, dset_name, test_fraction)
    assert(int(test_fraction*(data['tr_size']+data['ts_size'])) == data['ts_size'])
    assert(data['ts_size'] == len(ts_indexes))
    return load_dataset(dset_fname, ts_indexes)

def get_kfold_splits(exp_name, dset_name, n, k):
    kfold_fname, exists = get_fname_exp(exp_name, dset_name, indexes='kfold')
    if exists:
        kfold_splits = get_data(kfold_fname)
        assert(k == kfold_splits.n_folds)
        assert(n == kfold_splits.n)
        log('Previously computed cross-validation indexes (%d folds) loaded.' % k)
    else:
        log('Building cross-validation indexes (%d folds)...' % k)
        kfold_splits = cross_validation.KFold(n=n, n_folds=k)
        save_data(kfold_splits, kfold_fname)
    return kfold_splits
