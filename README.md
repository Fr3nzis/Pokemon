# Pokémon Battle Outcome Prediction

This repository provides a complete coede for trying to predict the winner of Pokémon Showdown battles. The project is designed for the Kaggle competition *fds-pokemon-battles-prediction-2025* and includes raw-data ingestion, feature engineering for battle logs, model training and hyperparameter tuning, stacking and voting ensembles, and  submission generation.

## Project Structure

- **fds-pokemon-battles-prediction-2025/**  
  Raw Kaggle dataset: `train.jsonl`, `test.jsonl`.

- **set_up_scripts/**  
  Modules for data ingestion, preprocessing, and feature construction.  
  - `data_processing.py`: loads JSONL battles, inspects raw logs, constructs model-ready datasets through `DataHandler` and `DataProcessor`.  
  - `dicts.py`: Pokémon type mappings and status penalty dictionaries.  
  - `pk_functions.py`: functions for damage statistics, switch counts, and type-matchup effectiveness.  
  - `features_ext.py`: full feature engineering pipeline for stacking/logistic workflows.  
  - `features_ext_vot.py`: variant of feature engineering for the voting workflow.  
  - `set_up.py`: runs data loading, feature generation, dataset construction for the stacking pipeline.  
  - `set_up_vot.py`: analogous to `set_up.py` but using the voting-specific feature set.

- **models/**  
  Training utilities, tuning logic, and ensemble implementations.  
  - `tuning_function.py`: GridSearchCV infrastructure and hyperparameter grids for logistic regression, KNN, decision tree, random forest, AdaBoost, and XGBoost. Saves best estimators and scaling objects.  
  - `tuned_models_generation.py`: performs tuning for selected models, writes optimized and saves results in **generated_models/**.  
  - `stacking_functions.py`: data loading helpers, scaling utilities, base-model loading, stacking classifier construction, tuning, evaluation, and prediction.  
  - `stacking_model_generation.py`: manages the full stacking workflow (load → scale → assemble → tune → validate → predict).  
  - `voting_model.py`: constructs and evaluates a voting ensemble from tuned base models, then generates final predictions.  
  - `logistic.py`: baseline logistic regression workflow using tuned logistic model and its scaling.  
  - **generated_models/**:  tuned models generated (logistic, KNN, decision tree, random forest, AdaBoost, XGBoost), their scalers, and the final stacking/voting models.

- **data/**  
  Processed datasets generated during execution (`df_train`, `df_test`, `X_train`, `X_test`, `Y_train`, in CSV and PKL format).

- **pipeline.py**  
  Complete stacking pipeline: raw-data processing, feature extraction, model tuning, stacking ensemble creation, logistic model evaluation, and submission generation.

- **pipeline_vot.py**  
  Complete voting pipeline: raw-data processing (voting feature set), model tuning, voting classifier creation, and submission generation.

## Workflow

### 1. Raw Data Loading
`DataHandler` reads `train.jsonl` and `test.jsonl`, parses each battle into Python dictionaries, performs structural checks, and exposes them to the feature-engineering modules. `inspect_first_battle()` provides a reference example of the battle timeline and metadata.

### 2. Feature Engineering
`FeatureHandler` converts each battle into a structured feature vector by aggregating:
- lead Pokémon base stats (HP, Attack, Defense, Special Attack, Special Defense, Speed);
- Pokémon types and type-effectiveness measures computed via `get_effectiveness`;
- damage summaries (total inflicted, received, HP swings, net damage) extracted from the battle timeline using `damage_features`;
- switching behaviour (difference between players) using `switch_difference`;
- accuracy and power aggregates from moves used in the battle;
- presence of status conditions using `status_penalties`;
- additional indicators characterising battle's events.

The result is exported as `df_train` and `df_test`.  
`DataProcessor` then produces:
- `X_train`, `X_test`: final feature matrices;
- `Y_train`: binary outcome for Player 1.

### 3. Hyperparameter Tuning
`tuning_function.py` defines the tuning logic for all supported models. For each algorithm, it:
- instantiates the estimator and parameter grid;
- scales features when required (e.g., logistic regression, KNN);
- performs grid search with cross-validation (`GridSearchCV`, `KFold`);
- saves the best estimator and scaler into `models/generated_models/`.

`tuned_models_generation.py` calls this routine for the models used by the ensemble workflows (logistic regression, XGBoost, and AdaBoost).

### 4. Stacking Ensemble
`stacking_functions.py` loads the tuned models, applies the appropriate scaler, and constructs a `StackingClassifier` with logistic regression as meta-learner.  
It provides:
- separate functions for model assembly and model tuning;
- cross-validated evaluation (mean accuracy, standard deviation);
- final training on the full dataset and test-set prediction.

`stacking_model_generation.py` runs the complete stacking pipeline and writes `submission_ST.csv`.

### 5. Logistic Baseline
`logistic.py` loads the tuned logistic regression model and evaluates it separately. It trains the logistic model on the full scaled training set and generates `submission_LOG.csv`.

### 6. Voting Ensemble
`voting_model.py` loads the tuned base models, scales the data using the shared scaler, builds a `VotingClassifier`, evaluates it through cross-validation, fits it on the full dataset, and generates `submission_VOT.csv`.

### 7. Pipelines
- `pipeline.py`: runs setup → tuning → stacking → logistic, producing both stacking and logistic submissions.  
- `pipeline_vot.py`: runs setup (voting feature set) → tuning → voting and produces the voting submission.

