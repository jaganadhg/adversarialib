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
from os.path import realpath, dirname, join, abspath, basename

def main(verbose=True):
    this_folder = dirname(realpath(__file__))
    code_folder = abspath(join(this_folder, '../'))
    sys.path.insert(0, code_folder)
    curr_working_dir = os.getcwd()
    
    from prlib.learn import save_learn_result
    from prlib.dataset import build_train_test_splits, get_kfold_splits
    from util import make_exp_folders, log
    
    parser = argparse.ArgumentParser(description='Builds all target Classifiers')
    parser.add_argument('exp_path', type=str, help='Experiment Folder (must contain a setup.py file)')
    args = parser.parse_args()
    
    exp_path = join(curr_working_dir,args.exp_path)
    exp_name = basename(exp_path)
    dset_folder = join(curr_working_dir,args.dset_folder)
        
    sys.path.insert(0,exp_path)
    from setup import DSET_FOLDER, DSET_NAME, CLASSIFIER_PARAMS, GRID_PARAMS, TEST_FRACTION, NFOLDS, NSPLITS
    
    log("Starting experiment %s." % exp_name)
    make_exp_folders(exp_path)
    
    data = build_train_test_splits(exp_path, dset_folder, DSET_NAME, TEST_FRACTION, NSPLITS)
    kfolds = get_kfold_splits(exp_path, DSET_NAME, data['tr_size'], NFOLDS)
    for class_type, params in CLASSIFIER_PARAMS.items():
        for key, val in GRID_PARAMS.items():
            params['grid_search'][key] = val
        params['grid_search']['cv'] = kfolds
        for split_no in range(NSPLITS): # TODO: MULTI-THREADING?
            save_learn_result(exp_path, dset_folder, DSET_NAME, TEST_FRACTION, split_no, class_type, params, verbose)

if __name__ == "__main__":
    main()
