import pandas as pd
import numpy as np 
import json
from scipy.io import loadmat
import sys

from whobpyt.datatypes import par, Recording
from whobpyt.models.RWW import RNNRWW, ParamsRWW
from whobpyt.optimization.custom_cost_RWW import CostsRWW
from whobpyt.run import Model_fitting
import optuna

# Normalise the FC data and find Correlation matrix
def normalise_correlate_fc(fc):
    fc_emp = fc / np.max(fc)
    fc_emp = np.corrcoef(fc_emp.T)
    return fc_emp

def get_data(DATA_PATH):
    healthy_patients = pd.read_csv(DATA_PATH)

    fcs = []

    for f in healthy_patients['FC_DATA']:
        fc = loadmat(f)
        fc = fc['ROI_activity'][:100, :]
        if fc.shape[1] == 197:
            fc = fc.T
            fcs.append(fc)

    return fcs

def get_avg_fc(fcs):
    corr_matrices = []

    for data in fcs:
        fc = normalise_correlate_fc(data)
        corr_matrices.append(fc)

    stacked_corr = np.stack(corr_matrices, axis = 0)
    avg_corr = np.mean(stacked_corr, axis =0)
    
    return avg_corr

def get_avg_fmr(fmris):
    stacked_corr = np.stack(fmris, axis = 0)
    avg_corr = np.mean(stacked_corr, axis =0)
    
    return avg_corr

def train_model(TPperWindow, step_size, repeat_size, tr, num_epochs, learning_rate, g, node_size=100):

    params = ParamsRWW(g=par(g, g, 1/np.sqrt(10), True, True), g_EE=par(3.5, 3.5, 1/np.sqrt(50)), g_EI =par(0.42, 0.42, 1/np.sqrt(50)), \
                   g_IE=par(0.42, 0.42, 1/np.sqrt(50)), I_0 =par(0.2), std_in=par(0.0), std_out=par(0.00))
    
    # call model want to fit
    model = RNNRWW(node_size, TPperWindow, step_size, repeat_size, tr, sc, True, params)
    
    # create objective function
    ObjFun = CostsRWW(model)

    # call model fit
    F = Model_fitting(model, ObjFun)

    # train model
    F.train(u = 0, empRecs = [data_mean], num_epochs = num_epochs, TPperWindow = TPperWindow, learningrate = learning_rate)

    #  predict FC
    ts_sim, fc_sim = F.simulate(u = 0, num_windows=int(data_mean.length/TPperWindow), TPperWindow = TPperWindow, base_window_num = 20)
    
    # evaluate performance
    fc_cor, _ = F.evaluate(u = 0, empRecs = [data_mean], TPperWindow = TPperWindow, fc_sims=[fc_sim], ts_sims=[ts_sim])

    return fc_cor

def objective(trial):
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.05, 0.1])
    num_epochs = trial.suggest_categorical('num_epochs', [20, 50, 100])
    step_size = trial.suggest_categorical('step_size', [0.01, 0.05, 0.1])
    g = trial.suggest_categorical('G', [80, 100, 500, 1000])
    score = train_model(TPperWindow=20, step_size=step_size, repeat_size=5, tr=tr, g=g, num_epochs=num_epochs, learning_rate=learning_rate)
    return score

if __name__ == '__main__':
    # ADNI_MERGE_PATH = 'data/CN_ADNIMERGE_29Apr2024_wFiles.csv'
    # SC_PATH = 'data/DTI_fiber_consensus_HCP.csv'
    study_name = sys.argv[1]
    ADNI_MERGE_PATH = sys.argv[2]
    SC_PATH = sys.argv[3]
    optuna_trials = int(sys.argv[4])
    no_jobs = int(sys.argv[5])

    fcs = get_data(ADNI_MERGE_PATH)
    sc = np.genfromtxt(SC_PATH, delimiter=',')

    tr = 0.75

    SC = (sc + sc.T) * 0.5
    sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

    fc_emp = get_avg_fc(fcs)

    # normalise all fcs
    for fc in fcs:
        fc = fc / np.max(fc)

    avg_fmri = get_avg_fmr(fcs)

    # prepare data structure of the model
    fMRIstep = tr
    data_mean = Recording(avg_fmri.T, fMRIstep)

    study = optuna.create_study(direction='maximize', study_name=study_name)  
    study.optimize(objective, n_trials=optuna_trials, n_jobs=no_jobs) 

    res = {
        'params': study.best_params,
        'score' : study.best_value
    }

    with open(f'optuna_{study.study_name}.json', "w") as file:
        json.dump(res, file)
    