#!/bin/bash

#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -o /rds/general/user/dc420/home/ad_modelling_fyp/logs/predicted_ventr/
#PBS -e /rds/general/user/dc420/home/ad_modelling_fyp/logs/predicted_ventr/

module load tools/prod
module load SciPy-bundle/2023.07-gfbf-2023a

cp -r $HOME/ad_modelling_fyp/whole_brain_models/whobpyt $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/requirements.txt $TMPDIR

pip install -r requirements.txt

cp $HOME/ad_modelling_fyp/data/ADNIMERGE/ADNIMERGE_29Apr2024_wFiles_mod.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/DTI_fiber_consensus_HCP.csv $TMPDIR

cp $HOME/ad_modelling_fyp/data/avg_scans/AB_CN.csv $TMPDIR
cp $HOME/ad_modelling_fyp/data/avg_scans/TAU_CN.csv $TMPDIR
cp $HOME/ad_modelling_fyp/predictors/gbregressor_ventricular_icv.pkl $TMPDIR
cp $HOME/ad_modelling_fyp/whole_brain_models/predict_from_sAB.py $TMPDIR

cp -r $HOME/FMRI_ADNI_DATA/fc $TMPDIR

python predict_from_sAB.py 

cp trialled_sAB_E.txt $HOME/ad_modelling_fyp/data/prediction_res
cp predicted_ventr_icv_sAB_E.txt $HOME/ad_modelling_fyp/data/prediction_res
cp predicted_ventr_icv_sAB_E_plot.png $HOME/ad_modelling_fyp/data/prediction_res