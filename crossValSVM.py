from os import listdir
from os.path import isfile, join
from random import shuffle
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import numpy
from operator import itemgetter
from time import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import learningCurve
import matplotlib.pyplot as plt
import os
from LBP import getFeature

allFolders = ['data100x100/' + dI for dI in os.listdir('data100x100') if os.path.isdir(os.path.join('data100x100', dI))]
paths = allFolders
allFiles = []
testNames = []
trainNames = []

for folderIndex, folder in enumerate(allFolders):
    onlyfiles = [f for f in listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for index,val in enumerate(onlyfiles):
        onlyfiles[index] = paths[folderIndex] + '/' + val
    #seventy = int(len(onlyfiles) * .7)
    trainNames += onlyfiles[:]
    #testNames += onlyfiles[seventy:]

trainImages = []
trainLabels = []
d = [1]
for path in trainNames:
    img = plt.imread(path)
    feature = getFeature(path).reshape(-1)
    trainImages.append(feature)
    trainLabels.append(path.split('/')[1])
trainImages = np.array(trainImages)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainImages, trainLabels, test_size=0.2, random_state=0)


def crossValidationSVC():
    parameters = {
        'C': [1,2,3],
        'kernel': ["linear", "poly", "rbf", "sigmoid"],
        'degree' : [1,2,3,4,5,6,7],
        'gamma':['auto'],
        'coef0':[1.0,2.0,3.0,4.0,5.0,6.0],
        'shrinking': [True, False],
        'probability':[True,False],
        'max_iter':[-1.0,100.0,200.0,500.0,100.0],
        'decision_function_shape':['ovo','ovr']
    }

    svc = SVC()

    ts_gs = run_gridsearch(X_train, y_train, svc, param_grid=parameters, cv=2)
    print("\n-- Best Parameters:")
    for k, v in ts_gs.items():
        print("parameter: {:<20s} setting: {}".format(k, v))



def run_gridsearch(X, y, clf, param_grid, cv):
    grid_search = GridSearchCV(clf,param_grid, cv = cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return  top_params

def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
              numpy.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters


if __name__ == '__main__':
    crossValidationSVC()
