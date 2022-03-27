import keras_tuner as kt
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import sklearn.pipeline ### REQUIRED FOR KERAS TUNER TO WORK! ###

import pdb

def build_model(hp):
    model_type = hp.Choice('model_type', ['random_forest', 'ridge'])
    if model_type == 'random_forest':
      model = ensemble.RandomForestClassifier(
#              learning_rate = hp.Float('learning_rate', 0.02, 1, step=10),
              n_estimators=hp.Int('n_estimators', 10, 50, step=10),
              max_depth=hp.Int('max_depth', 3, 10))
    else:
      model = linear_model.RidgeClassifier(
              alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
    return model

tuner = kt.tuners.SklearnTuner(
        oracle=kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=model_selection.StratifiedKFold(5),
        directory='.',
        project_name='my_project')

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)


tuner.search(X_train, y_train)

best_model = tuner.get_best_models(num_models=1)[0]
print(best_model)
pdb.set_trace()
