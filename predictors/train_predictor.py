'''
Train + Hyperparameter tune Random Forrest Model
'''
import pandas as pd
import sys
import numpy as np
from scipy.io import loadmat
import json

# scikit-learn modules
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.ensemble import RandomForestRegressor # for building the model
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score

def get_fc_data(df, FC_DATA_PATH):
    # Get the FC data as numpy arrays
    dim_x = len(df['FC_DATA'])
    features = np.zeros(shape=(dim_x, 100, 200)) # get the first 100 regions

    for i, file in enumerate(df['FC_DATA'].values):
        arr = loadmat(f'{FC_DATA_PATH}/{file}')['ROI_activity'][:100, :] # get the first 100 regions
        if arr.shape[1] != 200:
            # add padding to get a constant shape
            diff = 200 - arr.shape[1]
            if diff < 0:
                arr = arr[:, :200]
            else:
                pad_width = ((0, 0), (0, diff))  
                padded_array = np.pad(arr, pad_width, mode='constant', constant_values=0)
        features[i] = padded_array
    features_2d = features.reshape(features.shape[0], -1)
    return features_2d

def clean_data(x_train, x_test, y_train, y_test):
    # Remove NaNs in target
    y_train_cleaned = y_train.reset_index(drop=True)
    y_test_cleaned =y_test.reset_index(drop=True)

    nan_indices = y_train_cleaned.index[y_train_cleaned.isna()]
    y_train_cleaned = y_train_cleaned.drop(nan_indices)
    x_train_cleaned = np.delete(x_train, nan_indices, axis = 0)

    nan_indices_test = y_test_cleaned.index[y_test_cleaned.isna()]
    y_test_cleaned = y_test_cleaned.drop(nan_indices_test)
    x_test_cleaned = np.delete(x_test, nan_indices_test, axis = 0)

    return x_train_cleaned, x_test_cleaned,  y_train_cleaned, y_test_cleaned

def randomised_grid_search(x_train, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 100)]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()

    # Evaluation metric
    r2_scorer = make_scorer(r2_score)

    # Random search of parameters, using 5 fold cross validation, 

    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(
        estimator = rf, param_distributions = random_grid, scoring=r2_scorer,
        n_iter = 50, cv = 5, verbose=4, random_state=42, n_jobs = -1)

    # Fit the random search model
    rf_random.fit(x_train, y_train)

    # Get the best parameter set
    best_params = rf_random.best_params_

    print("Best parameters:", best_params)
    return best_params

def evaluate(x_test, y_test, model):
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print("R2 Score: ", r2)
    return r2

if __name__ == '__main__':
    ADSP_DATA_PATH = "../data/ADSP_PHC_COGN_Dec2023_FILTERED_wfiles.csv"
    PREDICTOR_TYPE = 'VSP'
    FC_DATA_PATH = '../FMRI_ADNI_DATA/fc'
    # ADSP_DATA_PATH = sys.argv[1]
    # PREDICTOR_TYPE = sys.argv[2]
    # FC_DATA_PATH = sys.argv[3]

    df = pd.read_csv(ADSP_DATA_PATH)
    print("Getting features and target...")
    features = get_fc_data(df, FC_DATA_PATH)
    targets = df[f'PHC_{PREDICTOR_TYPE}']
    print("Done")

    # split into test + training (80% train, 20% test)
    print("Splitting data into test and train sets...")
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 28)
    print("Done")

    print("Cleaning data...")
    x_train, x_test, y_train, y_test = clean_data(x_train, x_test, y_train, y_test)
    print("Done")

    # # Start Randomised Search
    # print("Starting randomised search...")
    # best_params = randomised_grid_search(x_train, y_train)

    # PARAM_FILE = f'{PREDICTOR_TYPE}_best_params.json'

    # # Write data to a JSON file
    # with open(PARAM_FILE, 'w') as json_file:
    #     json.dump(best_params, json_file)

    # best_rf = RandomForestRegressor(n_estimators=best_params['n_estimators'],
    #                                 max_features=best_params['max_features'],
    #                              max_depth=best_params['max_depth'],
    #                              min_samples_split=best_params['min_samples_split'],
    #                              min_samples_leaf=best_params['min_samples_leaf'],
    #                              bootstrap=best_params['bootstrap'],
    #                              random_state=42)

    # # Train the model with the best parameters
    # best_rf.fit(x_train, y_train)

    # # Evaluate

    