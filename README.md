# Code and data for "Towards a speech-based digital biomarker for cognitive decline" 

This repository contains code and data for the npj Digital Medicine submission entitled "Towards a speech-based digital biomarker for cognitive decline" (Heitz, Engler, Langer, 2025)

As the manuscript is currently in review, the repository might still change and will be cleaned until publication.

## Repository structure
- `conda/`: `environment.yaml` file for conda environment setup
- `data/`: Cognitive test results and speech-derived feature values of a subset of participants (n=837). See Data Availability statement in the manuscript for details.
- `src/data_preparation`: Code used in the preparation of the data from Prolific (e.g. automatic transcription)
- `src/data_analysis`: Data analysis pipeline code for data loading, feature extraction, regression and classification models

The main entry point for the analyses is `src/data_analysis/run/run.sh`: