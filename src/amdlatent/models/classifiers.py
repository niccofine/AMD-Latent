# PACKAGE IMPORTS

# System packages
import hashlib

# Machine learning packages
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# CUSTOM FUNCTIONS

def stablehash(*parts):
    """
    A 32-bit hash that stays identical across interpreter sessions, unlike Python's
    built-in hash() for str/bytes (salted per-process by PYTHONHASHSEED), so it is safe to
    derive a reproducible seed from.
        Inputs:  - [parts]: Any number of values, stringified and joined to form the hash key.
        Outputs: - [value: int]: A deterministic 32-bit integer.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

def fitseed(task, classifier, mode, repetition):
    """
    Deterministic per-fit seed for trainmodel's randomstate: hashes which (task, classifier,
    split mode, repetition) is being fit, so re-running the same configuration reproduces
    the same fit while different configurations (and different repetitions of the same
    configuration) draw independent seeds.
        Inputs:  - [task: str]: The classification task (e.g. "Binary No", "Binary Late", "All").
                 - [classifier: str]: The classifier name (as used by trainmodel).
                 - [mode: str]: The train/val split mode this fit belongs to (e.g. "Couples-wise", "Patient-wise") — or any other value distinguishing this fit from same-repetition fits under a different configuration (e.g. a hyperparameter, for a grid search with no repetition index of its own).
                 - [repetition: int]: Which repetition this fit is (0-indexed), or any other per-fit distinguishing value.
        Outputs: - [seed: int]: A seed in [0, 2**31 - 2], safe to pass as trainmodel's randomstate.
    """
    return stablehash(task, classifier, mode, repetition) % (2**31 - 1)

def trainmodel(X_train, y_train, modeltype, hyper, randomstate=None):
    if modeltype == 'Random Forest':
        model = RandomForestClassifier(n_estimators=hyper, random_state=randomstate)
    elif modeltype == 'SVM':
        model = SVC(probability=True,kernel='rbf', C=hyper[0], gamma=hyper[1], random_state=randomstate)
    elif modeltype == 'SVM Linear':
        model = SVC(probability=True,kernel='linear', C=hyper, random_state=randomstate)
    elif modeltype == 'AdaBoost':
        model = AdaBoostClassifier(n_estimators=hyper, random_state=randomstate)
    elif modeltype == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=hyper, max_iter=1000, random_state=randomstate)
    model.fit(X_train, y_train)
    return model