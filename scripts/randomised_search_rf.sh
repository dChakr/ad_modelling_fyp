#!/bin/bash

#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=50:mem=500gb

#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/randomised_search_rf/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/randomised_search_rf/

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a

cp $HOME/ad_modelling_fyp/data/ADSP_PHC_COGN_Dec2023_FILTERED_wfiles_forTEMP.csv $TMPDIR
cp $HOME/ad_modelling_fyp/train_predictor.py $TMPDIR
cp -r $HOME/FMRI_ADNI_DATA/fc $TMPDIR

python train_predictor.py $TMPDIR/ADSP_PHC_COGN_Dec2023_FILTERED_wfiles_forTEMP.csv $ARGUMENT $TMPDIR/fc

cp $ARGUMENT\_best_params.json $HOME/ad_modelling_fyp/data/
