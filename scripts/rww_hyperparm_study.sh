#!/bin/bash

#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=10:mem=500gb
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/optuna_studies/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/optuna_studies/

module load tools/prod
module load SciPy-bundle/2023.07-gfbf-2023a
pip install optuna

cp -r $HOME/ad_modelling_fyp/whobpyt $TMPDIR
cp $HOME/ad_modelling_fyp/requirements.txt $TMPDIR

pip install -r requirements.txt

cp $HOME/ad_modelling_fyp/data/CN_ADNIMERGE_29Apr2024_wFiles.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/DTI_fiber_consensus_HCP.csv $TMPDIR

cp $HOME/ad_modelling_fyp/train_rww.py $TMPDIR
cp -r $HOME/FMRI_ADNI_DATA/fc $TMPDIR

python train_rww.py $ARGUMENT 'CN_ADNIMERGE_29Apr2024_wFiles.csv' 'DTI_fiber_consensus_HCP.csv' 1 1

cp optuna_$ARGUMENT $HOME/ad_modelling_fyp/data/
