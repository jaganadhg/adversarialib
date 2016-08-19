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
import os, numpy, threading
from collections import deque
from util import log, get_fname_exp, get_classifiers, DATA, TESTS, BASE, ATTACK
from prlib.dataset import get_testing_data, load_dataset, get_indexes
from advlib.dataset import save_value, split_attacks
from sklearn.svm.classes import SVC
from prlib.classifier import MLP, get_score
from prlib.dataset import get_index_data
from sklearn.preprocessing import Normalizer
from gradient_mlp import gradient_mlp
from sklearn.metrics import pairwise
from scipy import sparse
from gradient_distances import *
from constraints import *
from multiprocessing import Pool, Manager, cpu_count


PREC = "%.4f"

def save_attack_iter(pattern, fname, sep=','):
    if type(pattern) == sparse.csr.csr_matrix:
        pattern = pattern.toarray()[0]
    save_value(" ".join([PREC % val for val in pattern]), fname, sep)

def save_boundary_info_threadsafe(lock, p_no, boundary_info, fname_data, fname_attack_score, fname_score, save_attack_patterns):
    boundary_info.sort(lambda x,y: cmp(x[0], y[0]))
    patterns = []
    attack_scores = []
    target_scores = []
    
    for bound_no, pattern, attack_score, target_score in boundary_info:
        if save_attack_patterns:
            patterns.append(" ".join([PREC % val for val in pattern]))
        # REAL score for the target classifier
        target_scores.append(PREC % target_score)
        # Score for the classifier known to (or built by) the adversary
        attack_scores.append(PREC % attack_score)
    
    with lock:     
        save_value("%d %s %s" % (p_no, " ".join(attack_scores), os.linesep), fname_attack_score)
        save_value("%d %s %s" % (p_no, " ".join(target_scores), os.linesep), fname_score)
    
        if save_attack_patterns:
            save_value("%d %s %s" % (p_no, ",".join(patterns), os.linesep), fname_data)

def gradient_svm_linear(classifier, pattern):
    return classifier.coef_[0]

def gradient_svm_rbf(classifier,pattern):
    grad = []
    dual_coef = classifier.dual_coef_
    support = classifier.support_vectors_
    gamma = classifier.get_params()['gamma']
    kernel = pairwise.rbf_kernel(support, pattern, gamma)
    for element in range(0,len(support)):
        if (grad == []):
            grad = (dual_coef[0][element]*kernel[0][element]*2*gamma*(support[element]-pattern))
        else:
            grad = grad + (dual_coef[0][element]*kernel[element][0]*2*gamma*(support[element]-pattern))
    return -grad #il bel Maiorca si e' dimenticato un meno! :)

def gradient_svm_poly(classifier, pattern):
    grad = []
    dual_coef = classifier.dual_coef_
    support = classifier.support_vectors_
    degree = classifier.get_params()['degree']
    R = classifier.get_params()['coef0']
    gamma = classifier.get_params()['gamma']
    kernel = pairwise.polynomial_kernel(support,pattern, degree-1, gamma, R)
    #log("Kernel: %s; Support: %s; Pattern: %s" % (kernel,support, pattern))
    #kernel = pairwise.rbf_kernel(support, pattern, gamma)
    for element in range(0, len(support)):
        if(grad == []):
            grad = dual_coef[0][element]*degree*kernel[element][0]*support[element]*gamma
        else:
            grad = grad + dual_coef[0][element]*degree*kernel[element][0]*support[element]*gamma
    return -grad

def evaluate_stop_criteria(obj_function_at_pattern, epsilon):
    #print obj_function_at_pattern
    if len(obj_function_at_pattern) < obj_function_at_pattern.maxlen:
        return False
    if (obj_function_at_pattern[0]-obj_function_at_pattern[-1]) < epsilon:
        return True
    return False

def update_candidate_root_patterns(candidate_root_patterns, within_boundary_constraints, **constraint_params):
    for index, candidate_root_pattern in enumerate(candidate_root_patterns):
        if not within_boundary_constraints(pattern=candidate_root_pattern, **constraint_params):
            return candidate_root_patterns[:index]
    return candidate_root_patterns

def compute_gradient(attack_classifier, pattern, leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params):
    grad = gradient(attack_classifier, pattern)
    if lambda_value > 0:
        closer_leg_patterns, grad_mimicry, dist = gradient_mimicry(pattern, leg_patterns, **mimicry_params)
        grad_update = grad+lambda_value*grad_mimicry
        
        #print numpy.linalg.norm(grad), numpy.linalg.norm(lambda_value*grad_mimicry)
        
    else:
        closer_leg_patterns = leg_patterns
        grad_update = grad
        dist = 0
    # BAT: devo controllare la norma, non i singoli elementi
    if (numpy.linalg.norm(grad_update) != 0):
        grad_update = grad_update/numpy.linalg.norm(grad_update)
        # TODO: aggiungere ramo "else".
        # se il gradiente e' zero siamo arrivati in un minimo locale e dobbiamo uscire.
        
    return closer_leg_patterns, dist, grad_update

def thread_task(lock, p_no, root_pattern, attack_classifier, max_boundaries, stop_criteria_window, 
                within_boundary_constraints, apply_boundary_constraints, constraint_params, leg_patterns, 
                maxiter, gradient, lambda_value, gradient_mimicry, mimicry_params, step, stop_criteria_epsilon, 
                target_classifier, fname_data, fname_attack_score, fname_score, save_attack_patterns):
    # Gradient Descent Attack in place...
    boundary_numbers = range(1,max_boundaries+1)
    boundary_numbers.reverse()
    candidate_root_patterns = [root_pattern]
    attacker_score =  get_score(attack_classifier, root_pattern)
    target_score =  get_score(target_classifier, root_pattern)
    boundary_info = [(0, root_pattern, attacker_score, target_score)]
    closer_leg_patterns, dist, grad_update = compute_gradient(attack_classifier, root_pattern, leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params) 
    
    for bound_no in boundary_numbers:
        obj_function_at_pattern = deque(maxlen=stop_criteria_window)
        candidate_root_patterns = update_candidate_root_patterns(candidate_root_patterns, within_boundary_constraints, root_pattern=root_pattern, bound_no=bound_no, **constraint_params)
        if not candidate_root_patterns:
            log('Attack pattern %d. No candidate root patterns for Bound no. %d. Gradient Descent terminated.' % (p_no, bound_no))
            break
        
        pattern = candidate_root_patterns[-1] # last pattern which satisfies the new boundary constraint 
        num_candidate_root_patterns = len(candidate_root_patterns)
        closer_leg_patterns, dist, grad_update = compute_gradient(attack_classifier, pattern, closer_leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params) 
        
        # initial values...
        attacker_score =  get_score(attack_classifier, pattern)
        target_score =  get_score(target_classifier, pattern)
        obj_fun_value = attacker_score+lambda_value*dist
        obj_function_at_pattern.append(obj_fun_value)
        
        # DEBUG
        if max_boundaries == 1:
            fdebug = open(fname_attack_score+'_DEBUG.txt', 'a')
            firstline = True
        ### end of DEBUG
        
        for iter_no in range(num_candidate_root_patterns, maxiter):
            new_pattern = apply_boundary_constraints(root_pattern, pattern, grad_update, step, bound_no, **constraint_params)
            
            # this grad_update will be used in the next iteration
            closer_leg_patterns, dist, new_grad_update = compute_gradient(attack_classifier, new_pattern, closer_leg_patterns, lambda_value, gradient, gradient_mimicry, **mimicry_params) 
            
            #print pattern, attack_classifier.decision_function(pattern)[0], target_classifiers_info[0][0].decision_function(pattern)[0]
            #raw_input('iter: %d, next?' % iter_no)
            new_attacker_score =  get_score(attack_classifier, new_pattern)
            obj_fun_value = new_attacker_score+lambda_value*dist
            
            #print (obj_fun_value, dist, obj_fun_value-obj_function_at_pattern[-1], iter_no)
            
            # here we evaluate whether we reached a local minimum/there is a bouncing effect due to the chosen step value
            if obj_fun_value == obj_function_at_pattern[-1]:
                # stop criteria is not evaluated if obj_fun does not change at all...
                log('Attack pattern: %d, Boundary: %d, Iteration: %d. Local Minimum Reached. Obj.: %f' % (p_no, bound_no, iter_no, obj_fun_value))
                # NB: in this case, we do not update attacker_score and target_score
                break
            elif obj_fun_value < obj_function_at_pattern[-1]:
                obj_function_at_pattern.append(obj_fun_value)
            else:
                # it means that the step is too large to decrement the obj function...
                # in the next interation (discrete spaces) we skip the feature that has been changed
                for idx, elm in enumerate(pattern==new_pattern):
                    if not elm:
                        log('Attack pattern: %d, Boundary: %d, Jumping off the minimum: feature %d will be skipped at iteration %d' % (p_no, bound_no, idx, iter_no))
                        grad_update[idx] = 0
                        break
                continue
            
            # DEBUG
            if max_boundaries == 1:
                c = new_pattern-pattern
                idx = numpy.where(c!=0)[0][0]
                if firstline:
                    firstline = False
                else:
                    fdebug.write(',')
                fdebug.write('%d %d %d %.4f' % (iter_no, idx, numpy.sign(c[idx]), dist))
                # END OF DEBUG
            
            # if we are here: no local minimum has been reached yet, thus we can compute the new_target_score 
            new_target_score =  get_score(target_classifier, new_pattern)
            
            # special case: we store all iterations if max_boundaries == 1
            if max_boundaries == 1:
                boundary_info.append((iter_no, new_pattern, new_attacker_score, new_target_score))
            
            if evaluate_stop_criteria(obj_function_at_pattern, stop_criteria_epsilon):
                log('Attack pattern: %d, Boundary: %d, Iteration: %d. Stop criteria reached. Obj.: %f' % (p_no, bound_no, iter_no, obj_function_at_pattern[-1]))
                break
            
            if not (new_pattern==candidate_root_patterns[-1]).all():
                candidate_root_patterns.append(new_pattern)
            
            # we update all values for the next iteration
            pattern = new_pattern
            grad_update = new_grad_update
            attacker_score = new_attacker_score
            target_score = new_target_score
        
        #print p_no, bound_no, obj_function_at_pattern[-1]
        # DEBUG
        if max_boundaries == 1:
            fdebug.write('\n')
            fdebug.close()
        # END OF DEBUG
        
        if num_candidate_root_patterns < maxiter:
            if iter_no == maxiter-1:
                log('Attack pattern: %d, Boundary: %d, Maxiter reached. Obj. value: %f' % (p_no, bound_no, obj_function_at_pattern[-1]))
        else:
            log('Attack pattern: %d, Boundary: %d, Maxiter <= Number of previously computed attack points within the new boundary (no new attack iterations performed).' % (p_no, bound_no))
        
        # if max_boundaries > 1 we store only patterns that reached a local minimum/maximum iterations 
        if max_boundaries > 1:
            boundary_info.append((bound_no, pattern, attacker_score, target_score))
    save_boundary_info_threadsafe(lock, p_no, boundary_info, fname_data, fname_attack_score, fname_score, save_attack_patterns)


#===============================================================================
# NOTE: Here we assume that attacks normally show a higher score with respect to
# legitimate samples. Attack Label=1, Legitimate Label=-1 The default decision 
# threshold is 0. That is, the evasion attack is considered successful if an 
# attack pattern receives a score < 0.
#===============================================================================
def gradient_descent_attack(attack_patterns, leg_patterns, attack_classifier, attack_classifier_type,  
                            fname_attack_score, fname_data, target_classifier, fname_score, norm_weights=None,
                            lambda_value=0.5, mimicry_distance='euclidean', mimicry_params={},
                            save_attack_patterns=False, maxiter=500, score_threshold=0, step=0.15,
                            stop_criteria_window = 5, stop_criteria_epsilon = 10**(-4),
                            constraint_function=None, max_boundaries=1, threads=4, constraint_params={}):
        
    # SELECTION OF THE ATTACK TECHNIQUE ACCORDING TO THE CLASSIFIER BUILT BY THE ATTACKER
    # IDEALLY, THIS IS THE REAL CLASSIFIER    
    attack_classifier_type = type(attack_classifier)
    if attack_classifier_type == SVC and attack_classifier.kernel == 'linear':
        gradient = gradient_svm_linear
    elif attack_classifier_type == SVC and attack_classifier.kernel == "rbf":
        gradient = gradient_svm_rbf
    elif attack_classifier_type == SVC and attack_classifier.kernel == "poly":
        gradient = gradient_svm_poly
    elif attack_classifier_type == MLP:
        gradient = gradient_mlp
    else:
        log("Gradient Descent Attack: unsupported attack classifier %s." % attack_classifier_type)
        return

    # SELECTION OF THE MIMICRY TECHNIQUE ACCORDING TO THE CHOSEN DISTANCE MEASURE
    if mimicry_distance == 'euclidean':
        gradient_mimicry = gradient_euclidean_dist
    elif mimicry_distance == 'kde_euclidean':
        gradient_mimicry = gradient_kde_euclidean_dist
    elif mimicry_distance == 'kde_hamming':
        gradient_mimicry = gradient_kde_hamming_dist
    else:
        log("Gradient Descent Attack: unsupported mimicry distance %s." % mimicry_distance)
        return
     
    if not constraint_function:
        apply_boundary_constraints = apply_no_constraints
        within_boundary_constraints = within_no_constraints
        max_boundaries = 1 # we overwrite the value... it does not make sense to have multiple boundaries
    elif constraint_function == 'box':
        apply_boundary_constraints = apply_hypercube
        within_boundary_constraints = within_hypercube
    elif constraint_function == 'hamming':
        apply_boundary_constraints = apply_hamming
        within_boundary_constraints = within_hamming
    elif constraint_function == 'only_increment':
        apply_boundary_constraints = apply_only_increment
        within_boundary_constraints = within_only_increment
        num_features = len(attack_patterns[0])
        if norm_weights is None:
            constraint_params['weights'] = numpy.ones(num_features)/step
        else:
            assert(len(norm_weights)==num_features)
            constraint_params['weights'] = norm_weights
            constraint_params['feature_upper_bound'] = 1
            log('Norm weights loaded successfully from setup!')
        mimicry_params['weights'] = constraint_params['weights']
        constraint_params['inv_weights'] = numpy.array([1/item for item in constraint_params['weights']])
    else:
        log("Gradient Descent Attack: unsupported constraint function %s." % constraint_function)
        return
    
    if threads < 0:
        threads = cpu_count() # all cpus are employed... :-D
    
    if threads > 1:
        log("Gradient Descent will employ %d concurrent processes." % threads)
        manager = Manager()
        lock = manager.Lock()
        pool = Pool(threads)
        for p_no, root_pattern in enumerate(attack_patterns):
            pool.apply_async(func=thread_task, 
                        args=(lock, p_no, root_pattern, attack_classifier, max_boundaries, stop_criteria_window, 
                        within_boundary_constraints, apply_boundary_constraints, constraint_params, leg_patterns, 
                        maxiter, gradient, lambda_value, gradient_mimicry, 
                        mimicry_params, step, stop_criteria_epsilon, target_classifier, fname_data, 
                        fname_attack_score, fname_score, save_attack_patterns))
        pool.close()
        pool.join()
    else: # just to be able to easily block execution by keyboard interrupt... :D
        from threading import Lock
        lock = Lock()
        for p_no, root_pattern in enumerate(attack_patterns):
            thread_task(lock, p_no, root_pattern, attack_classifier, max_boundaries, stop_criteria_window, 
                        within_boundary_constraints, apply_boundary_constraints, constraint_params, leg_patterns, 
                        maxiter, gradient, lambda_value, gradient_mimicry, 
                        mimicry_params, step, stop_criteria_epsilon, target_classifier, fname_data, 
                        fname_attack_score, fname_score, save_attack_patterns)
#    lock = threading.Lock()
#    if threads > 1:
#        pool = ThreadPool(threads)
#        for p_no, root_pattern in enumerate(attack_patterns):
#            pool.add_task(thread_task, lock, p_no, root_pattern, attack_classifier, max_boundaries, stop_criteria_window, 
#                        update_candidate_root_patterns, within_boundary_constraints, apply_boundary_constraints, 
#                        constraint_params, leg_patterns, maxiter, gradient, lambda_value, gradient_mimicry, 
#                        mimicry_params, step, stop_criteria_epsilon, target_classifier, fname_data, 
#                        fname_attack_score, fname_score, save_attack_patterns)
#        pool.wait_completion()
#    else: # just to be able to easily block execution by keyboard interrupt... :D
#        for p_no, root_pattern in enumerate(attack_patterns):
#            thread_task(lock, p_no, root_pattern, attack_classifier, max_boundaries, stop_criteria_window, 
#                        update_candidate_root_patterns, within_boundary_constraints, apply_boundary_constraints, 
#                        constraint_params, leg_patterns, maxiter, gradient, lambda_value, gradient_mimicry, 
#                        mimicry_params, step, stop_criteria_epsilon, target_classifier, fname_data, 
#                        fname_attack_score, fname_score, save_attack_patterns)
        
def gradient_descent_run(attack_classifier, attack_classifier_type, fname_attack_score, fname_dset, fname_indexes,
                         fname_data, target_classifier, fname_score, attack_class, training, relabeling, **kwargs):
    # REAL Dataset
    if fname_indexes:
        indexes = get_indexes(fname_indexes)
    else:
        indexes = None
    X, y = load_dataset(fname_dset, indexes)
    fname_score_exists = os.path.exists(fname_attack_score)
    fname_data_exists = os.path.exists(fname_data)
    

    # TODO: a more time-saving solution?
    if fname_score_exists:
        os.unlink(fname_attack_score)
    
    # TODO: a more time-saving solution?
    if kwargs['save_attack_patterns'] and fname_data_exists:
        os.unlink(fname_data)
    
    X_attack, y_attack, X_benign, y_benign  = split_attacks(X, y, attack_class)
    if relabeling:
        y_target = target_classifier.predict(X)
        a, b, X_benign, y_benign  = split_attacks(X, y_target, attack_class)
    
    gradient_descent_attack(X_attack, X_benign, attack_classifier, attack_classifier_type, 
                            fname_attack_score, fname_data, target_classifier, fname_score, **kwargs)
    

def single_attack(X, y, classifier, attack_classifier, exp_name, dset_name, attack_class_name, attack_class_type, target_class_type, split_no, attack_class, relabeling, **kwargs):
    log('%s is employed to attack %s... Go Forrest GO!!' % (attack_class_type, target_class_type))
            
    fname_attack_score, fname_attack_score_exists = get_fname_exp(exp_name, dset_name, feed_type=TESTS, exp_type=ATTACK,
                            attack_type = 'gradient_descent', class_type=attack_class_name, split_no=split_no)
    fname_data, fname_data_exists = get_fname_exp(exp_name, dset_name, feed_type=DATA, exp_type=ATTACK,
                            attack_type = 'gradient_descent', class_type=attack_class_name, split_no=split_no, split_type='ts')
    fname_score, fname_score_exists = get_fname_exp(exp_name, dset_name, feed_type=TESTS, exp_type=ATTACK,
                            attack_type = 'gradient_descent', class_type='REAL_'+attack_class_name, split_no=split_no)
    
    if fname_score_exists or fname_data_exists or fname_attack_score_exists:
        log("It seems that such a test has been already performed.")
        return
    
    X_attack, y_attack, X_benign, y_benign  = split_attacks(X, y, attack_class)
    if relabeling:
        y_target = classifier.predict(X)
        a, b, X_benign, y_benign  = split_attacks(X, y_target, attack_class)

    gradient_descent_attack(X_attack, X_benign, attack_classifier, attack_class_type, 
                            fname_attack_score, fname_data, classifier, fname_score, **kwargs)

def gradient_descent(classifier, class_type, exp_path, dset_folder, dset_name, test_fraction, split_no, attack_class, training, relabeling, 
                     fname_metachar_attack_vs_target, fname_metachar_samples_repetitions, **kwargs):
    
    # REAL Dataset
    X, y = get_testing_data(exp_path, dset_folder, dset_name, test_fraction, split_no)
    
    attack_classifiers = get_classifiers(exp_path, ATTACK, 'gradient_descent')
    
    # conoscenza perfetta
    single_attack(X, y, classifier, classifier, exp_path, dset_name, class_type+fname_metachar_attack_vs_target+class_type, class_type, class_type, split_no, attack_class, relabeling, **kwargs)      
    
    for attack_class_name in attack_classifiers.keys():
        attack_classifier = attack_classifiers[attack_class_name][split_no]
        attack_class_type, target_class_type = attack_class_name.split(fname_metachar_attack_vs_target)
        if target_class_type == class_type:
            single_attack(X, y, classifier, attack_classifier, exp_path, dset_name, attack_class_name, attack_class_type, target_class_type, split_no, attack_class, relabeling, **kwargs)
    
