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
from util import log, get_fname_exp, save_data, check_fname, CLASSIFIERS, ATTACK
from prlib.dataset import get_testing_data
from prlib.learn import learn
from random import sample
from sklearn import cross_validation
from scipy import sparse
import os

def save_learn_result_base(X, y, indexes, output_folder, nfolds, params, class_type, fname_metachar_samples_repetitions, verbose):
    size = len(X)
    dset_knowledge = params['dataset_knowledge']
    for n in dset_knowledge['samples_range']:
        params['grid_search']['cv'] = cross_validation.KFold(n=n, n_folds=nfolds)
        for rep_no in range(dset_knowledge['repetitions']):
            indexes = sample(xrange(size),n)
            class_name = ''.join((class_type, fname_metachar_samples_repetitions, str(n), fname_metachar_samples_repetitions, str(rep_no)))
            class_fname = os.path.join(output_folder, class_name)
            fname_classifier, exists = check_fname(class_fname)
            if exists:
                log("It seems that classifier %s has been already built." % class_name)
                return
            
            clf = learn(X[indexes], y[indexes], class_type, params)
            
            if verbose:
                for par, mean_score, scores in clf.grid_scores_:
                    print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, par)
            log("Best score: %s, Best Parameters: %s" % (clf.best_score_, clf.best_params_))
         
            save_data(clf.best_estimator_, fname_classifier)
            log('%s - The "best" classifier has been saved on %s' % (class_name, fname_classifier))

def save_learn_result(exp_path, dset_folder, dset_name, test_fraction, split_no, class_type, target_classifier, 
                      target_class_type, relabeling, nfolds, params, fname_metachar_attack_vs_target, 
                      fname_metachar_samples_repetitions, verbose):
    X, y = get_testing_data(exp_path, dset_folder, dset_name, test_fraction, split_no)
    if relabeling:
        y = target_classifier.predict(X)
        
    size = len(X)
    dset_knowledge = params['dataset_knowledge']
    for n in dset_knowledge['samples_range']:
        params['grid_search']['cv'] = cross_validation.KFold(n=n, n_folds=nfolds)
        for rep_no in range(dset_knowledge['repetitions']):
            indexes = sample(xrange(size),n)
            class_name = ''.join((class_type, fname_metachar_samples_repetitions, 
                                  str(n), fname_metachar_samples_repetitions, str(rep_no), 
                                  fname_metachar_attack_vs_target, target_class_type))
            
            fname_classifier, exists = get_fname_exp(exp_path, dset_name, feed_type=CLASSIFIERS, exp_type=ATTACK, 
                                                     attack_type='gradient_descent', class_type=class_name, split_no=split_no, 
                                                     split_type='ts')
            if exists:
                log("%s - dataset split: %s - It seems that a classifier has been already built." % (class_name, split_no))
                return
            
            log("%s - dataset split: %s - Finding the best parameters through cross-validation." % (class_name, split_no))
            
            clf = learn(X[indexes], y[indexes], class_type, params)
            
            if verbose:
                for par, mean_score, scores in clf.grid_scores_:
                    print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, par)
            log("Best score: %s, Best Parameters: %s" % (clf.best_score_, clf.best_params_))
         
            save_data(clf.best_estimator_, fname_classifier)
            log('%s - dataset split: %s - The "best" classifier has been saved on %s' % (class_name, split_no, fname_classifier))
