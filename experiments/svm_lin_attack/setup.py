import numpy as np
from os.path import realpath, dirname, join, abspath # new
from util import BASE

this_setup_folder = dirname(realpath(__file__))
DSET_FOLDER = abspath(join(this_setup_folder, '../../dataset'))
DSET_NAME = 'norm_pdf_med'

TEST_FRACTION = 0.15
NFOLDS = 3
NSPLITS = 3
CLASSIFIER_PARAMS = {
	'SVM_lin': {
		'lib': 'sklearn.svm.SVC',
		'common': {'kernel': 'linear'},
		'grid_search': {'param_grid': dict(C=np.logspace(-3, 2, 6))},
	},
}

GRID_PARAMS = {'iid': True, 'n_jobs': 1}

NORM_WEIGHTS_FILEPATH = join(DSET_FOLDER, 'norm_weights.txt')
NORM_WEIGHTS = np.array([float(item) for item in open(NORM_WEIGHTS_FILEPATH).read().split()])

ATTACK_CLASSIFIER_PARAMS = CLASSIFIER_PARAMS

ATTACK_PARAMS = {
	'gradient_descent': {
		'main_routine': 'prlib.test.classification_test',
		'main_routine_params': {'test_fraction':0.15, 'exp_type': BASE},
		'attack': {
			'attack_class': 1, 'maxiter': 500,
			'score_threshold': 0, 'step':0.01,
			'fname_metachar_attack_vs_target': '@',
			'fname_metachar_samples_repetitions': '-',
			'norm_weights': NORM_WEIGHTS,
			'constraint_function': 'only_increment', 'constraint_params': {'only_increment_step': 1},
			'stop_criteria_window': 20, 'stop_criteria_epsilon': 10**(-9), 'max_boundaries': 10,
			'lambda_value':10, 'mimicry_distance': 'kde_hamming', 'relabeling': True,
			'mimicry_params' : {'max_leg_patterns': 100 , 'gamma': 0.001},
			'save_attack_patterns': True, 'threads': 1,
			'training': {
				'dataset_knowledge': {'samples_range': range(50,100,50), 'repetitions': 1},
				'classifier_params': ATTACK_CLASSIFIER_PARAMS,
			},
		},
	},
}