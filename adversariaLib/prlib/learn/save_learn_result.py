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
from util import log, get_fname_exp, save_data, CLASSIFIERS, BASE
from prlib.dataset import get_training_data
from learn import learn

def save_learn_result(exp_name, dset_folder, dset_name, test_fraction, split_no, class_type, params, verbose):
    X, y = get_training_data(exp_name, dset_folder, dset_name, test_fraction, split_no)
    
    fname_classifier, exists = get_fname_exp(exp_name, dset_name, feed_type=CLASSIFIERS, exp_type=BASE,
                                            class_type=class_type, split_no=split_no, split_type='tr')
    if exists:
        log("%s - dataset split: %s - It seems that a classifier has been already built." % (class_type, split_no))
        return
    
    log("%s - dataset split: %s - Finding the best parameters through cross-validation." % (class_type, split_no))
    
    clf = learn(X, y, class_type, params)
    
    if verbose:
        for par, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, par)
    log("Best score: %s, Best Parameters: %s" % (clf.best_score_, clf.best_params_))
 
    save_data(clf.best_estimator_, fname_classifier)
    log('%s - dataset split: %s - The "best" classifier has been saved on %s' % (class_type, split_no, fname_classifier))
