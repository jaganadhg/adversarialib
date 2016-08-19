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

def save_value(value, fname, sep=' '):
    f = open(fname, "a")
    f.write("%s%s" % (sep, value))
    f.close()

def split_attacks(X, y, attack_class):
    output = X_attack, y_attack, X_benign, y_benign = [], [], [], []
    for i, pattern in enumerate(X):
        if y[i] == attack_class:
            X_attack.append(pattern)
            y_attack.append(y[i])
        else:
            X_benign.append(pattern)
            y_benign.append(y[i])
    return tuple([np.array(elm) for elm in output])
