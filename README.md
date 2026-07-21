# Overtraining and Injury: Predicting Athlete Injury Risk from Performance and Training Data

Data analysis and machine learning project that identifies overtraining-related problems and predicts injury risk in athletes from performance, physical, and training metrics.

## Project Overview

Overtraining injuries are a major challenge for athletes at every level. This project uses four datasets to answer three questions:

1. How common are overtraining-related problems?
2. Can specific training variables predict injury?
3. Can overtraining-related injuries be predicted accurately?

After cleaning the data and training a Random Forest Classifier, the model reaches a high-accuracy baseline that can help athletes and coaches spot structural injury risk earlier.

## Datasets Used

Four CSV files tracking subject demographics, activities, and mechanical markers:

- `High_Accuracy_Sport_Injury_Dataset.csv` — primary dataset for the predictive model
- `run_data_meta.csv`
- `walk_data_meta.csv`
- `wlinj_dryad.csv`

## Data Pipeline & Cleaning

Athletic data often has missing values. Preprocessing steps:

1. **Threshold dropping** — Drop any column missing more than 40% of its values.
2. **Imputation** — Fill remaining numeric NaNs with `0`; label categorical missing values as `"Unknown"`.
3. **Feature scaling** — Apply `MinMaxScaler` so physical attributes and training metrics share equal weight.

## Model Performance

A Random Forest Classifier was trained with an 80/20 train-test split.

- **Model:** `RandomForestClassifier(n_jobs=-1, random_state=42)`
- **Features:** Age, Gender, Height_cm, Weight_kg, BMI, Training Frequency, Training Duration, Warmup Time, Sleep Hours, Flexibility Score, Muscle Asymmetry, Recovery Time, Injury History, Stress Level, Training Intensity
- **Target:** `Injury_Risk` (0 or 1)

### Results

**Accuracy = 0.97**

97% accuracy on held-out test data, which makes the model a strong baseline for early overtraining risk signals.

## Dependencies

Install the Python libraries used by the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipython
```

## How to Use

1. Clone the repository.
2. Keep the four CSV files in the same directory as the notebook.
3. Open and run `injury_predict.ipynb` to run the full pipeline from cleaning through model evaluation.

A related presentation is included as `Presentation_ACSC.pdf`.
