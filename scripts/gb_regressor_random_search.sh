#!/bin/bash

#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=10:mem=50gb
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/gb_search/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/gb_search/

module load tools/prod
module load SciPy-bundle/2023.07-gfbf-2023a

cp $HOME/ad_modelling_fyp/data/ADNIMERGE_29Apr2024_Ventricles_ICV_mod.csv $TMPDIR

cp $HOME/ad_modelling_fyp/predictors/ventricular_icv_random_search.py $TMPDIR
cp -r $HOME/FMRI_ADNI_DATA/fc $TMPDIR

python ventricular_icv_random_search.py 'ADNIMERGE_29Apr2024_Ventricles_ICV_mod.csv' $ARGUMENT

cp random_search_gbreg_ab_tau_$ARGUMENT.json $HOME/ad_modelling_fyp/data/
