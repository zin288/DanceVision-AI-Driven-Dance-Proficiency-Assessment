from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize,  LabelBinarizer
from sklearn.pipeline import make_pipeline
import argparse
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, roc_auc_score
import numpy as np

import warnings

"""
This file will run through a number of scikit learn models on the the training data
in training.csv.  This training data was collected through running:
"""

# model_name = 'best_ymca_pose_model'
model_name = 'supershy_model_12pt'


def get_data(file_name):
    """
    read training.csv and return the X,y as series
    :return: X - the data representing the road view
             y - what turn value
    """
    df = pd.read_csv(f'{file_name}', header=None)
    # print(df.head())
    X = df.loc[:, 1:]
    y = df.loc[:, 0]
    # print(X.shape)
    # print(y.shape)
    classes = []
    if y.dtype == object:
        # then we need to labelbinarize it
        le = LabelEncoder()
        y_notused = le.fit_transform(y)
        classes = le.classes_
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_val, y_train, y_val, classes


def train_model(model, X_train, X_val, y_train, y_val, name=None, param_grid=None):

    metrics = {'Model Name': name}

    if name:
        print(f"Training: {name}")

    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X_train, y_train)

        metrics['Training Score (CV)'] = grid.best_score_

        print(grid.best_score_)
        print(grid.best_params_)
        print(grid.best_estimator_)
        _best_model = grid.best_estimator_
        _best_params = grid.best_params_
        _best_score = grid.best_score_
    else:
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)
        print('cv scores', cv_scores, '\nMean cv scores:',cv_scores.mean())
        metrics['Training Score (CV)'] = cv_scores.mean()
        _best_model = model
        _best_params = param_grid
        _best_score = cv_scores.mean()

    # Evaluate on the validation set
    y_pred = _best_model.predict(X_val)
    y_prob = _best_model.predict_proba(X_val)
    metrics['Accuracy'] = accuracy_score(y_val, y_pred)
    metrics['Precision'] = precision_score(y_val, y_pred, average='micro')
    metrics['Recall'] = recall_score(y_val, y_pred, average='micro')
    metrics['F1 Score'] = f1_score(y_val, y_pred, average='micro')
    metrics['Log Loss'] = log_loss(y_val, y_prob)

    # # Print y_pred, y_prob, and y_val before calculating ROC AUC
    # print("Predictions (y_pred): ", y_pred[:10]) # print first 10 predictions
    # print("Probabilities (y_prob): ", y_prob[:10]) # print first 10 probabilities
    # print("Actual values (y_val): ", y_val[:10]) # print first 10 actual values

    # Binarize the y_val labels
    lb = LabelBinarizer()
    y_val_bin = lb.fit_transform(y_val)

    try:
        if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
            roc_auc = roc_auc_score(y_val_bin, y_prob)
        else:
            roc_auc = roc_auc_score(y_val_bin, y_prob[:, 1], average='micro', multi_class="ovr")
        metrics['ROC AUC'] = roc_auc
    except ValueError:
        metrics['ROC AUC'] = 'Not applicable'   
    
    return _best_score, _best_params, _best_model, metrics


def create_logistic_regression_model():
    logreg = LogisticRegression(multi_class='multinomial')
    return logreg


def create_decision_tree():
    tree = DecisionTreeClassifier()
    return tree


def create_svc():
    svc = SVC(kernel='linear', C=1, probability=True)
    return svc


def create_gnb():
    gnb = GaussianNB()
    return gnb


def create_knn():
    knn = KNeighborsClassifier()
    return knn


def create_linear():
    lin = LinearRegression()
    return lin


def find_best_model(X_train, X_val, y_train, y_val):
    models = [
        {
            'model': make_pipeline(StandardScaler(), create_logistic_regression_model()),
            'params_grid': dict(logisticregression__penalty=['l2'], logisticregression__C=[10, 1, 0.1, 0.01], logisticregression__solver=['newton-cg', 'sag', 'lbfgs'],
                                logisticregression__max_iter=[100, 200, 300]),
            'name': 'LogisticRegression',
            'skip': False
        }
        # ,
        # {
        #     'model': make_pipeline(StandardScaler(), create_decision_tree()),
        #     'params_grid': dict(decisiontreeclassifier__criterion=['gini', 'entropy'], decisiontreeclassifier__max_depth=[2, 3, 4, 5], decisiontreeclassifier__min_samples_split=[2, 3]),
        #     'name': 'DecisionTree',
        #     'skip': False
        # }
        # ,
        # # -added probability=True in create_svc()
        # {
        #     'model': make_pipeline(StandardScaler(), create_svc()),
        #     'params_grid': dict(svc__kernel=['linear', 'rbf', 'poly'], svc__gamma=['auto', 'scale']),
        #     'name': 'SVC',
        #     'skip':True
        # }
        # ,
        
        # {
        #     'model': make_pipeline(StandardScaler(), create_gnb()),
        #     'params_grid': None,
        #     'name': 'GaussianNB',
        #     'skip': True
        # }
        # ,
        # {
        #     'model': make_pipeline(StandardScaler(), create_knn()),
        #     'params_grid': dict(kneighborsclassifier__n_neighbors=list(range(1, 10)), 
        #                  kneighborsclassifier__weights=['uniform', 'distance']),
        #     'name': 'KNN GridSearch',
        #     'skip': False #True
        # }
        # ,
        
        # {
        #     'model': make_pipeline(StandardScaler(), create_knn()),
        #     'params_grid': None,
        #     'name': 'KNN Default',
        #     'skip': True
        # }
        # ,
        
        # {
        #     'model': make_pipeline(StandardScaler(), create_linear()),
        #     'params_grid': None,
        #     'name': 'Linear',
        #     'skip':True
        # }
        # ,
        # {
        #     'model': make_pipeline(StandardScaler(), RandomForestClassifier()),
        #     'params_grid': dict(randomforestclassifier__n_estimators=[100], randomforestclassifier__max_depth=[2,3,4]),
        #     'name': 'RandomForestClassifier',
        #     'skip': False
        # }
        # ,
        # {
        #     'model': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        #     'params_grid': None,
        #     'name': 'GradientBoostingClassifier',
        #     'skip': False #True

        # }
        # ,
        # {
        #     'model': make_pipeline(StandardScaler(), MLPClassifier()),
        #     'params_grid': dict(mlpclassifier__activation=['relu'],
        #             mlpclassifier__solver=['sgd', 'adam'],
        #             mlpclassifier__alpha=[100, 10, 1], 
        #             mlpclassifier__max_iter=[500, 600],
        #             mlpclassifier__hidden_layer_sizes=[(X_train.shape[1], 128, 16), (X_train.shape[1], 100)]),
        #     'name': 'MLP',
        #     'skip': True
        # }

    ]
    best_model = None
    best_params = None
    best_score = -1


    metrics_list = []


    for model in models:
        if not model['skip']:
            score, params, best, metrics = train_model(model['model'], X_train, X_val, y_train, y_val, name=model['name'], param_grid=model['params_grid'])
            metrics_list.append(metrics)

            if metrics['Training Score (CV)'] > best_score:
                best_params = metrics  # Store all metrics as best parameters for now
                best_model = best
                best_score = metrics['Training Score (CV)']

    # Convert list of dictionaries to DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Print the DataFrame
    print(metrics_df)

    return best_model, best_params, best_score, metrics_df


'''
python 02_pose_model_training.py --training-data ymca_training.csv --model-name ymca_pose_model

'''
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-data", type=str, required=False, default='training1.csv',
                    help="name of the training data file")
    ap.add_argument("--model-name", type=str, required=False, default=f'{model_name}',
                    help=f"name of the saved pickled model [no suffix]. Default: {model_name}.pkl")
    args = vars(ap.parse_args())

    model_name = args['model_name']
    training_data_filename = args['training_data']
    training_data_path = './data/' + training_data_filename

    X_train, X_val, y_train, y_val, classes = get_data(training_data_path)
  

    best_model, best_params, best_score, metrics_df = find_best_model(X_train, X_val, y_train, y_val)

    print("*******  Best Model and Parameters  *********")
    print('best_model:', best_model)
    print('best_params:', best_params)
    print('best_score:', best_score)
    print(metrics_df)
    with open(f'model/{model_name}_metadata.txt', 'w') as f:
        f.write(f'{best_model}\n')
        f.write(f'{best_params}\n')
        f.write(f'{best_score}\n')
        f.write(f'{metrics_df}\n')

    with open(f'model/{model_name}_classes.txt', 'w') as f:
        f.write(f"{classes}")


    joblib.dump(best_model, f"model/{model_name}.pkl")

    print(f"Done saving model to best model:  model/{model_name}.pkl")
    

