# Dataset Source

**Name:** Diabetes Health Indicators Dataset (BRFSS 2015)
**File used:** `diabetes_012_health_indicators_BRFSS2015.csv` (21.7 MB)

**Kaggle:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
**Original source:** US Centers for Disease Control and Prevention, Behavioral Risk
Factor Surveillance System (BRFSS) 2015.
https://www.cdc.gov/brfss/annual_data/annual_2015.html

## Contents

- 253,680 survey respondents
- 21 predictor features + 1 target = 22 columns
- Target `Diabetes_012`: 0 = no diabetes, 1 = prediabetes, 2 = diabetes
- Class counts: 213,703 / 4,631 / 35,346  (84.24% / 1.83% / 13.93%)
- Format: CSV, all columns numeric

## The other two files on Kaggle

The Kaggle page also publishes `diabetes_binary_health_indicators_BRFSS2015.csv`
and `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`. Neither is used.
The 50/50 file was rejected because its balance comes from discarding most
non-diabetic records, which leaves its predicted probabilities uncalibrated
against any real population. Section 2.6 of the report explains this.

## Licence

Kaggle dataset: CC0 (public domain). The underlying BRFSS survey is a CDC
public-use file.
