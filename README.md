# Modelling Neurodegeneration: Investigating Changes in Connectomics of Alzheimer’s Patients

Repository to store code associated with my final year project: "Modelling Neurodegeneration: Investigating Changes in Connectomics of Alzheimer’s Patients".

## Notable directories:
- `data`: contains the datasets used throughut this project, not including the fMRI subject data. Also includes the outcomes of hyperparameter tuning trials
- `helpers`: contains helper notebooks and scripts for pre-processing data sets and imaging data to make it useable by other code in this codebase
- `predictors`: contains all of the regressor models trialled during the development of the clinical measure predictor
- `scripts`: includes bash scripts needed to run several of the python scripts in `predictors` and `whole_brain_models` on Imperial's HPCs
- `whole_brain_models`: contains the code for developing the A\beta-Tau DMF model, hyperparameter tuning, and developing the adapted whole-brain modelling pipeline. Most importantly, `whole_brain_models/pipeline_models` contains the fitted models used in the final adapted pipeline, and `whole_brain_models/ad_modelling_pipeline.ipynb` contains the code for experiments of the overall pipeline

## Dependencies:

To run this repository, the libraries Nilearn, Ants-Py and Whobpyt are required. The former two can be install using `pip install <library>`, whereas Whobpyt needs to be cloned from [https://github.com/GriffithsLab/whobpyt/tree/main](https://github.com/GriffithsLab/whobpyt/tree/main).