# Overtraining and Injury: Predicting Athlete Injury Risk from Performance and Training Data
This repository contains a data analysis and machine learning project aimed at identifying overtraining-related problems and predicting injury risk in athletes using performance, physical, and training metrics.
## Project Overview
Injuries caused by overtraining are a significant hurdle for athletes across all levels. This project explores four distinct datasets to answer three central questions:
How common are overtraining-related problems?
Are specific training variables able to predict injury?
Is it possible to accurately predict these overtraining-related injuries?
By utilizing data cleaning techniques and implementing a Random Forest Classifier, this model achieves a high-accuracy predictive baseline to help athletes and coaches mitigate structural injury risks.
## Datasets Used
The project processes data from four source CSV files tracking subject demographics, activities, and mechanical markers:
High_Accuracy_Sport_Injury_Dataset.csv (Primary dataset used for the predictive model),
run_data_meta.csv,
walk_data_meta.csv,
wlinj_dryad.csv
## Data Pipeline & Cleaning
Real-world athletic data often suffers from missing metrics. The following preprocessing steps were taken to ensure clean training data:
Threshold Dropping: Evaluated missing data percentages across features. Any column missing more than 40% of its data was permanently dropped.
Imputation: For remaining missing data (NaN), numerical missing values were imputed with 0, and categorical columns were flagged as "Unknown".
Feature Scaling: Applied a MinMaxScaler to normalize individual physical attributes and training metrics, ensuring equal weight distribution across features.
## Model Performance
A Random Forest Classifier was trained using an 80/20 train-test split.
Model: RandomForestClassifier(n_jobs=-1, random_state=42)
Key Features Included: Age, Gender, BMI, Training Frequency, Training Duration, Warmup Time, Sleep Hours, Flexibility Score, Muscle Asymmetry, Recovery Time, and Injury History.
Prediction Target: Injury_Risk (0 or 1)
### Results
Accuracy=0.97
The model demonstrates an outstanding 97% accuracy rate on unseen test data, making it highly reliable for identifying early indicators of athlete overtraining risk.
## Dependencies
To run the notebook locally, ensure you have the following Python libraries installed:
Bash
pip install pandas numpy matplotlib scikit-learn ipython
## How to Use
Clone the repository to your local machine.
Ensure the four required CSV datasets are in the same directory as your script/notebook.
Run the Jupyter Notebook to execute the pipeline from data cleaning through model valuation.
