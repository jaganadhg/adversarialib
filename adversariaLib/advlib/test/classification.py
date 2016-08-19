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
from util import log, get_classifiers, BASE, ATTACK

def cross_classification_test(exp_name, dset_name, test_fraction, exp_type, function, **kwargs):
    routine = function.__name__
    log("Testing classifiers built for experiment %s (exp type: %s). Routine: %s" % (exp_name, exp_type, routine))
        
    attack_classifiers = get_classifiers(exp_name, ATTACK)
    classifiers = get_classifiers(exp_name, BASE)
    if not attack_classifiers:
        log("No attack classifiers found. We will " % (exp_name, exp_type))
        return
    if not classifiers:
        log("Did you run experiment %s (exp type: %s)? No classifiers found." % (exp_name, exp_type))
        return
    
    for class_type in classifiers.keys():
        for split_no in classifiers[class_type]:
            classifier = classifiers[class_type][split_no]
            log('Classifier %s (split_no: %s) under test...' % (class_type, split_no))
            function(classifier=classifier, class_type=class_type, split_no=split_no, 
                     exp_name=exp_name, dset_name=dset_name, test_fraction=test_fraction, **kwargs)
                
