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
import argparse, os, sys
from prlib.learn import save_learn_result
from prlib.dataset import build_train_test_splits, get_kfold_splits
from prlib.test import classification_test, save_test_result
from util import BASE, make_exp_folders, import_object, log

def main(verbose=True):
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('exp_path', type=str, help='Experiment Folder (must contain a setup.py file)')
    args = parser.parse_args()
    
    currdir = os.getcwd()
    
    exp_path = args.exp_path
    exp_name = os.path.basename(exp_path)
    if not args.exp_path.startswith('/'):
        exp_path = os.path.join(currdir,exp_path)
        
    sys.path.insert(0,exp_path)
    from setup import DSET_FOLDER, DSET_NAME, CLASSIFIER_PARAMS, GRID_PARAMS, ATTACK_PARAMS, TEST_FRACTION, NFOLDS, NSPLITS
    
    log("Starting experiment %s." % exp_name)
    make_exp_folders(exp_path)
    
    #===========================================================================
    # Build all Target classifiers
    #===========================================================================
    data = build_train_test_splits(exp_path, DSET_FOLDER, DSET_NAME, TEST_FRACTION, NSPLITS)
    kfolds = get_kfold_splits(exp_path, DSET_NAME, data['tr_size'], NFOLDS)
    for class_type, params in CLASSIFIER_PARAMS.items():
        for key, val in GRID_PARAMS.items():
            params['grid_search'][key] = val
        params['grid_search']['cv'] = kfolds
        for split_no in range(NSPLITS): # TODO: MULTI-THREADING?
            save_learn_result(exp_path, DSET_FOLDER, DSET_NAME, TEST_FRACTION, split_no, class_type, params, verbose)
            
    #===========================================================================
    # Standard Accuracy Test against all Target Classifiers
    #===========================================================================
    classification_test(exp_path, DSET_FOLDER, DSET_NAME, TEST_FRACTION, BASE, save_test_result, verbose=verbose)
    
    
    #===========================================================================
    # Build all Gradient Descent Classifiers
    #===========================================================================
    from advlib.attacks.gradient_descent import learn
    from util import get_classifiers
    
    attack_params = ATTACK_PARAMS['gradient_descent']['attack']
    fname_metachar_attack_vs_target = attack_params['fname_metachar_attack_vs_target']
    fname_metachar_samples_repetitions = attack_params['fname_metachar_samples_repetitions']
    training_params = attack_params['training']
    classifier_params = training_params['classifier_params']
    relabeling = attack_params['relabeling']
    log("Starting Learn Gradient Descent for experiment %s." % exp_name)
    target_classifiers = get_classifiers(exp_path, BASE)
    
    for class_type, params in classifier_params.items():
        for key, val in GRID_PARAMS.items():
            params['grid_search'][key] = val
        params['dataset_knowledge'] = training_params['dataset_knowledge']
        for split_no in range(NSPLITS): # TODO: MULTI-THREADING?
            for target_class_type in target_classifiers.keys():
                target_classifier = target_classifiers[target_class_type][split_no]
                learn.save_learn_result(exp_path, DSET_FOLDER, DSET_NAME, TEST_FRACTION, split_no, class_type, 
                                        target_classifier, target_class_type, relabeling, NFOLDS, params, 
                                        fname_metachar_attack_vs_target, fname_metachar_samples_repetitions, verbose)
    
    
    #===========================================================================
    # Attack
    #===========================================================================
    for attack_fun_str, params in ATTACK_PARAMS.items():
        main_routine = import_object(params['main_routine'])
        attack_fun = import_object("advlib.attacks." + attack_fun_str)
        attack_params = params['attack']
        attack_params.update(params['main_routine_params'])
        main_routine(exp_path=exp_path, dset_folder=DSET_FOLDER, dset_name=DSET_NAME, function=attack_fun, **attack_params)
    
    
if __name__ == "__main__":
    main()
