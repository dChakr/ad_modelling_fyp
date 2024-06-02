#!/bin/bash

#PBS -l walltime=56:00:00
#PBS -l select=1:ncpus=10:mem=50gb
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/rww_abt_trials/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/rww_abt_trials/

module load tools/prod
module load SciPy-bundle/2023.07-gfbf-2023a
pip install optuna

cp -r $HOME/ad_modelling_fyp/whole_brain_models/whobpyt $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/requirements.txt $TMPDIR

pip install -r requirements.txt

cp $HOME/ad_modelling_fyp/data/ADNIMERGE/ADNIMERGE_29Apr2024_wFiles_mod.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/DTI_fiber_consensus_HCP.csv $TMPDIR

cp $HOME/ad_modelling_fyp/data/avg_scans/AB_$PGROUP.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/avg_scans/TAU_$PGROUP.csv $TMPDIR

cp $HOME/ad_modelling_fyp/whole_brain_models/train_rww_abt.py $TMPDIR
cp -r $HOME/FMRI_ADNI_DATA/fc $TMPDIR

python train_rww_abt.py $STUDY_NAME $PGROUP $NO_TRIALS $NO_JOBS

cp optuna_$STUDY_NAME.json $HOME/ad_modelling_fyp/data/optuna_trials
