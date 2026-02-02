# Code and data for "Towards a speech-based digital biomarker for cognitive impairment: speech as a proxy for cognitive assessment" 

This repository contains code and data for the npj Digital Medicine paper:

Heitz, J., Engler, I.M. & Langer, N. Towards a speech-based digital biomarker for cognitive impairment: speech as a proxy for cognitive assessment. _npj Digit. Med._ (2026). https://doi.org/10.1038/s41746-026-02360-8



## Repository structure
- `analyses/`: Jupyter notebooks for the analyses of results, figures and tables in the manuscript
- `conda/`: `environment.yaml` file for conda environment setup
- `data/`: Cognitive test results and speech-derived feature values of a subset of participants (n=837). See Data Availability statement in the manuscript for details.
- `src/data_preparation`: Code used in the preparation of the data from Prolific (e.g. automatic transcription, data quality checks)
- `src/data_analysis`: Data analysis pipeline code for data loading, feature extraction, regression and classification models

Before running, update the constants in `src/config/constants.py`. 
The main entry point for the analyses is `src/data_analysis/run/run.sh`.

Note: The code currently depends on the original data. 
To reproduce the results on the subset of participants in `/data`, the code needs to be changed to load this data, 
which has as different structure.