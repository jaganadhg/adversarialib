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
from os.path import realpath, dirname, join, abspath, basename

def main(verbose=True):
    this_folder = dirname(realpath(__file__))
    code_folder = abspath(join(this_folder, '../'))
    sys.path.insert(0, code_folder)
    curr_working_dir = os.getcwd()
    
    from advlib.attacks.gradient_descent import gradient_descent_run
    from util import log, get_data
    
    parser = argparse.ArgumentParser(description='Attack a set of real classifiers using a single test set and a set of classifiers built by the adversary.')
    parser.add_argument('setup_path', type=str, help='Folder containing the setup file')
    parser.add_argument('input_file', type=str, help='Test set file path.') 
    parser.add_argument('attack_classifier', type=str, help="Attack Classifier path")
    parser.add_argument('target_classifier', type=str, help="Target Classifier path")
    parser.add_argument('output_folder', type=str, help='Scores and attack iterations folder')
    parser.add_argument('--index_file', type=str, help='Indexes file path (if any).', required=False)
    args = parser.parse_args()
    
    sys.path.insert(0,join(curr_working_dir,args.setup_path))
    from setup import CLASSIFIER_PARAMS, ATTACK_PARAMS
    
    attack_params = ATTACK_PARAMS['gradient_descent']['attack']
    del attack_params['fname_metachar_attack_vs_target']
    del attack_params['fname_metachar_samples_repetitions']
    attack_classifier = get_data(join(curr_working_dir,args.attack_classifier))
    attack_classifier_name = basename(args.attack_classifier)
    target_classifier_name = basename(args.target_classifier)
    target_classifier = get_data(join(curr_working_dir,args.target_classifier))
    if args.index_file:
        index_file = join(curr_working_dir,args.index_file)
    else:
        index_file = None
    fname_attack_score = join(curr_working_dir,args.output_folder, ".".join(["attack_scores", attack_classifier_name, "txt"]))
    fname_data = join(curr_working_dir,args.output_folder, ".".join(["attack_patterns", attack_classifier_name, "txt"]))
    fname_score = join(curr_working_dir,args.output_folder, ".".join(["real_scores", attack_classifier_name, target_classifier_name, "txt"]))
    gradient_descent_run(attack_classifier, attack_classifier_name, fname_attack_score, args.input_file, index_file, 
                        fname_data, target_classifier, fname_score, **attack_params)
        
if __name__ == "__main__":
    main()
