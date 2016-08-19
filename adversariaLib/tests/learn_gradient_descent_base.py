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
import os, argparse, sys
import numpy as np
from os.path import realpath, dirname, join, abspath, basename

def main(verbose=True):
    this_folder = dirname(realpath(__file__))
    code_folder = abspath(join(this_folder, '../'))
    sys.path.insert(0, code_folder)
    curr_working_dir = os.getcwd()
    
    from advlib.attacks.gradient_descent.learn import save_learn_result_base
    from prlib.dataset import load_dataset
    from prlib.learn import learn
    from util import log, save_data, get_data
     
    parser = argparse.ArgumentParser(description='Build a set of gradient descent attack classifiers from a single training set')
    parser.add_argument('setup_path', type=str, help='Folder containing the setup file')
    parser.add_argument('input_file', type=str, help='Training set file path.')
    parser.add_argument('output_folder', type=str, help='Output folder (each classifier will be stored here)')
    parser.add_argument('--target_classifier', metavar='T', type=str,
                        help='Target Classifier (for dataset re-labeling)')
    parser.add_argument('--nfolds', metavar='N', type=int,
                        help='Number of folds used for cross-validation.',
                        default=3)
    parser.add_argument('--indexes', metavar='I', type=str,
                        help='File containing pattern indexes (only those patterns will be employed for learning)')
    parser.add_argument('--verbose', metavar='V', type=bool,
                        help='Verbose?',
                        default=False)
    args = parser.parse_args()
    sys.path.insert(0,join(curr_working_dir,args.setup_path))
    from setup import ATTACK_PARAMS, GRID_PARAMS
    
    log("Learning from %s." % args.input_file)
    X, y = load_dataset(join(curr_working_dir,args.input_file))
    
    if args.target_classifier:
        log("Re-labeling dataset patterns employing classifier %s..." % args.target_classifier)
        target_classifier = get_data(join(curr_working_dir,args.target_classifier))
        y = target_classifier.predict(X)
    
    if args.indexes:
        indexes = join(curr_working_dir,args.indexes)
    else:
        indexes = None
    
    output_folder = join(curr_working_dir,args.output_folder)
    
    attack_params = ATTACK_PARAMS['gradient_descent']['attack']
    training_params = attack_params['training']
    fname_metachar_samples_repetitions = attack_params['fname_metachar_samples_repetitions']
    classifier_params = training_params['classifier_params']
    
    log("Attack classifiers: %s" % classifier_params.keys())
    
    for class_type, params in classifier_params.items():
        for key, val in GRID_PARAMS.items():
            params['grid_search'][key] = val
        params['dataset_knowledge'] = training_params['dataset_knowledge']
        save_learn_result_base(X, y, indexes, output_folder, args.nfolds, params, class_type, fname_metachar_samples_repetitions, verbose)

if __name__ == "__main__":
    main()
