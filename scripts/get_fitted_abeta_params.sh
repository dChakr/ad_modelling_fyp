#!/bin/bash

#PBS -l walltime=00:40:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/abt_fittes_params/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/abt_fittes_params/

module load tools/prod
module load SciPy-bundle/2023.07-gfbf-2023a

cp -r $HOME/ad_modelling_fyp/whole_brain_models/whobpyt $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/requirements.txt $TMPDIR

pip install -r requirements.txt

cp $HOME/ad_modelling_fyp/data/ADNIMERGE/ADNIMERGE_29Apr2024_wFiles_mod.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/DTI_fiber_consensus_HCP.csv $TMPDIR

cp $HOME/ad_modelling_fyp/data/avg_scans/AB_$PGROUP.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/avg_scans/TAU_$PGROUP.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/optuna_trials/optuna_$PGROUP\_study_2.json $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/get_fitted_abeta_vals.py $TMPDIR
cp -r $HOME/FMRI_ADNI_DATA/fc $TMPDIR

python get_fitted_abeta_vals.py $PGROUP

cp bAB_E_$PGROUP.txt $HOME/ad_modelling_fyp/data/fitted_params
cp sAB_E_$PGROUP.txt $HOME/ad_modelling_fyp/data/fitted_params
cp bAB_I_$PGROUP.txt $HOME/ad_modelling_fyp/data/fitted_params
cp sAB_I_$PGROUP.txt $HOME/ad_modelling_fyp/data/fitted_params
cp bt_E_$PGROUP.txt $HOME/ad_modelling_fyp/data/fitted_params
cp st_E_$PGROUP.txt $HOME/ad_modelling_fyp/data/fitted_params
