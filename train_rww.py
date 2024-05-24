import pandas as pd
import numpy as np 
import json
from scipy.io import loadmat
import sys
from sklearn.model_selection import train_test_split

from whobpyt.datatypes import par, Recording
from whobpyt.models.RWW import RNNRWW, ParamsRWW
from whobpyt.optimization.custom_cost_RWW import CostsRWW
from whobpyt.run import Model_fitting
import optuna
import torch

def get_data(DATA_PATH):
    healthy_patients = pd.read_csv(DATA_PATH)

    fcs = []

    for f in healthy_patients['FC_DATA']:
        # when you're not running this with a script, remember to add ../FMRI_ADNI_DATA/ before the filenames
        fc = loadmat(f)
        fc = fc['ROI_activity'][:100, :]
        if fc.shape[1] == 197:
            fc = fc.T
            fcs.append(fc)
        
    # normalise all fcs - zscore
    for i, fc in enumerate(fcs):
        fcs[i] = (fc - fc.mean(axis=0)) / fc.std(axis=0)
    
    # get training and test datasets
    train_data, test_data = train_test_split(fcs, test_size=0.2, random_state=42)

    return train_data, test_data

def get_avg_fc(fcs):
    corr_matrices = []

    for data in fcs:
        fc = np.corrcoef(data.T)
        corr_matrices.append(fc)

    stacked_corr = np.stack(corr_matrices, axis = 0)
    avg_corr = np.mean(stacked_corr, axis =0)
    
    return avg_corr

def get_avg_fmr(fmris):
    stacked_corr = np.stack(fmris, axis = 0)
    avg_corr = np.mean(stacked_corr, axis =0)
    
    return avg_corr

def train_model(fc_emp_train, fc_emp_test, TPperWindow, step_size, repeat_size, tr, num_epochs, learning_rate, g, ts_length, node_size=100):

    params = ParamsRWW(g=par(g, g, 1/np.sqrt(10), True, True), g_EE=par(3.5, 3.5, 1/np.sqrt(50)), g_EI =par(0.42, 0.42, 1/np.sqrt(50)), \
                   g_IE=par(0.42, 0.42, 1/np.sqrt(50)), I_0 =par(0.2), std_in=par(0.0), std_out=par(0.00))
    
    # call model want to fit
    model = RNNRWW(node_size, TPperWindow, step_size, repeat_size, tr, sc, True, params)
    
    # create objective function
    ObjFun = CostsRWW(model)

    # call model fit
    F = Model_fitting(model, ObjFun)

    # Train on training set
    F.train(u = 0, empFcs= [fc_emp_train], num_epochs = num_epochs, num_windows = int(ts_length / TPperWindow), learningrate = learning_rate)

    # Test on test set
    _, fc_sim = F.simulate(u = 0, num_windows=int(ts_length/TPperWindow), base_window_num = 20)

    fc_cor = F.evaluate(empFcs=[fc_emp_test], fc_sims=[fc_sim])

    return fc_cor

def objective(trial):
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.05, 0.1])
    num_epochs = trial.suggest_categorical('num_epochs', [20, 50, 100])
    step_size = trial.suggest_categorical('step_size', [0.01, 0.05, 0.1])
    g = trial.suggest_categorical('G', [20, 50, 80, 100, 500, 1000])
    tp_per_window = trial.suggest_categorical('TPperWindow', [5, 10, 20, 50])
    score = train_model(fc_emp_train=torch.from_numpy(fc_emp_train), fc_emp_test=fc_emp_test, TPperWindow=tp_per_window, step_size=step_size, repeat_size=5, tr=tr, g=g, num_epochs=num_epochs, ts_length=ts_length, learning_rate=learning_rate)
    return score

if __name__ == '__main__':
    # ADNI_MERGE_PATH = 'data/CN_ADNIMERGE_29Apr2024_wFiles.csv'
    # SC_PATH = 'data/DTI_fiber_consensus_HCP.csv'
    study_name = sys.argv[1]
    ADNI_MERGE_PATH = sys.argv[2]
    SC_PATH = sys.argv[3]
    optuna_trials = int(sys.argv[4])
    no_jobs = int(sys.argv[5])

    # normalise structural connectivity matrix
    sc = np.genfromtxt(SC_PATH, delimiter=',')

    SC = (sc + sc.T) * 0.5
    sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

    # get functional connectivity data
    train_data, test_data = get_data(ADNI_MERGE_PATH)

    tr = 0.75
    ts_length = train_data[0].shape[0]

    fc_emp_train = get_avg_fc(train_data)
    fc_emp_test = get_avg_fc(test_data)

    study = optuna.create_study(direction='maximize', study_name=study_name)  
    study.optimize(objective, n_trials=optuna_trials, n_jobs=no_jobs) 

    res = {
        'params': study.best_params,
        'score' : study.best_value
    }

    with open(f'optuna_{study.study_name}.json', "w") as file:
        json.dump(res, file)
    