"""
Experiment of predicting the CN ventricular volume score when varying a single dynamic parameter
"""
from whobpyt.datatypes import par
from whobpyt.models.RWWABT import RNNRWWABT, ParamsRWWABT
from whobpyt.optimization.custom_cost_RWW import CostsRWW
from whobpyt.run import Model_fitting
import pickle

# array and pd stuff
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

def define_model(fitted_sc, fitted_bAB_E, fitted_sAB_E, fitted_bt_E, fitted_st_E, fitted_bAB_I, fitted_sAB_I):
    params = ParamsRWWABT(bAB_E=par(val=fitted_bAB_E, fit_par=True), sAB_E=par(val=fitted_sAB_E, fit_par=True), bt_E=par(val=fitted_bt_E, fit_par=True),
                          st_E=par(val=fitted_st_E, fit_par=True), bAB_I=par(val=fitted_bAB_I, fit_par=True), sAB_I=par(val=fitted_sAB_I, fit_par=True))
    
    node_size = 100
    TPperWindow = 20
    step_size = 0.1
    repeat_size = 5
    tr = 0.75

    model = RNNRWWABT(node_size, TPperWindow, step_size, repeat_size, tr, fitted_sc, abeta, tau, True, params)

    ObjFun = CostsRWW(model)

    F = Model_fitting(model, ObjFun)
    
    return F

def simulate_FC(F):
    num_windows=9   # arbitrary choice - same as training
    _, fc_sim = F.simulate(u =0, num_windows=num_windows, base_window_num=20)
    return fc_sim

def compute_fc_lower_triangle(fc, node_size=100):
    # Get the lower triangle
    mask_e = np.tril_indices(node_size, -1)
    lower_triangle = fc[mask_e]
    
    return lower_triangle

def plot_predictions(sAB_Es, prediced_vs, filename):
    plt.figure(figsize=(8,6))
    sns.regplot(x=sAB_Es, y=prediced_vs, line_kws={'color':'red'}, scatter_kws={'s':10})
    plt.title('Predicted Ventricular_ICV Value from CN Model Simulations, Varying both sAB_E and sAB_I (CN SC Matrix)')
    plt.xlabel('sAB_E Value')
    plt.ylabel('Ventricular_ICV')
    plt.savefig(filename)
    

if __name__ == '__main__':
    # assume that these files are in the directory we're working in (adapted to be)
    ABETA_PATH = f'AB_CN.csv'
    TAU_PATH = f'TAU_CN.csv'
    PREDICTOR_PATH = 'gbregressor_ventricular_icv.pkl'   # '../predictors/gbregressor_ventricular_icv.pkl'
    CN_MODEL_PATH = 'cn_abt_model.pkl'
    AD_MODEL_PATH = 'AD_abt_model.pkl'

    with open(PREDICTOR_PATH, 'rb') as f:
        predictor = pickle.load(f)

    with open(CN_MODEL_PATH, 'rb') as f:
        cn_model = pickle.load(f)
    
    with open(AD_MODEL_PATH, 'rb') as f:
        ad_model = pickle.load(f)

    # get fitted values from CN model
    cn_fitted_sc = cn_model.model.sc_fitted.detach().numpy()
    ad_fitted_sc = ad_model.model.sc_fitted.detach().numpy()

    cn_fitted_bAB_E = cn_model.trainingStats.fit_params['bAB_E'][-1]
    # cn_fitted_sAB_E = cn_model.trainingStats.fit_params['sAB_E'][-1]
    cn_fitted_bt_E = cn_model.trainingStats.fit_params['bt_E'][-1]
    cn_fitted_st_E = cn_model.trainingStats.fit_params['st_E'][-1]
    cn_fitted_bAB_I = cn_model.trainingStats.fit_params['bAB_I'][-1]
    # cn_fitted_sAB_I = cn_model.trainingStats.fit_params['sAB_I'][-1]

    # get Abeta + Tau values for CN group
    abeta_file = np.genfromtxt(ABETA_PATH, delimiter=",")
    abeta = torch.tensor(abeta_file, dtype=torch.float32)

    tau_file = np.genfromtxt(TAU_PATH, delimiter=",")
    tau = torch.tensor(tau_file, dtype=torch.float32)

    # Set up experiment
    np.random.seed(42)  # Setting the seed for reproducibility
    sAB_Es = np.random.uniform(low=2.45, high=9.17, size=200)
    sAB_Es = np.sort(sAB_Es)
    sAB_Is = np.random.uniform(low=-11, high=-4.88, size=200)
    sAB_Is = np.sort(sAB_Is)[::-1]

    param_pairs = list(zip(sAB_Es, sAB_Is))

    prediced_vs = []

    for (sAB_E, sAB_I) in param_pairs:
        param_set = {
            'fitted_sc': cn_fitted_sc,
            'fitted_bAB_E': cn_fitted_bAB_E, 
            'fitted_sAB_E': sAB_E, 
            'fitted_bt_E': cn_fitted_bt_E, 
            'fitted_st_E': cn_fitted_st_E, 
            'fitted_bAB_I': cn_fitted_bAB_I, 
            'fitted_sAB_I': sAB_I
        }

        model = define_model(**param_set)
        simulations = []

        # find the average accross sims
        for i in range(5):
            fc = simulate_FC(model)
            lower_triangle_fc = compute_fc_lower_triangle(fc)
            simulations.append(lower_triangle_fc)
            
        v = predictor.predict(simulations)
        prediced_vs.append(np.mean(v))
    
    prediced_vs = np.array(prediced_vs)
    np.savetxt('trialled_sAB_I_for_both_CN_sc.txt', sAB_Is)
    np.savetxt('trialled_sAB_E_for_both_CN_sc.txt', sAB_Es)
    np.savetxt('predicted_ventr_icv_sAB_E_and_I_CN_sc.txt', prediced_vs)

    plot_predictions(sAB_Es, prediced_vs, 'predicted_ventr_icv_sAB_E_and_I_CN_sc_plot.png')