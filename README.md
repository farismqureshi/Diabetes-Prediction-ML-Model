# Diabetes Prediction ML Model

Predicting diabetes risk from behavioural and demographic health indicators — no blood test required.

A machine learning screening tool built on the CDC BRFSS 2015 health survey (253,680 respondents). The goal is not to diagnose, but to decide **who is worth testing first** in a population where testing everybody isn't affordable.

---

## The Problem

Type 2 diabetes is one of the most common chronic conditions worldwide, and a large share of people who have it don't know. Confirming a diagnosis requires a laboratory blood test — HbA1c or fasting plasma glucose — which costs money, requires a clinic visit, and simply isn't available to much of the population.

But the information needed to *guess* who is at risk — age, weight, blood pressure history, general health, physical activity — can be collected in a few minutes with no medical equipment at all.

**The question:** given only non-laboratory, self-reported health and lifestyle information, can a model reliably identify who is likely to have diabetes?

---

## Results

Final model: **LightGBM**, evaluated on a held-out test set of 45,957 records that played no part in training or model selection.

- ROC-AUC — 0.814
- Recall — 0.773 (77.3% of diabetic respondents correctly identified)
- PR-AUC — 0.471
- Precision — 0.349
- F1 — 0.481
- Balanced accuracy — 0.736
- Accuracy — 0.711

**On accuracy:** a model that answered "no diabetes" to everyone would score 82.7% accuracy while finding not a single diabetic person. Accuracy is the wrong metric here, and every model in this project deliberately scores below that baseline in exchange for actually catching cases. Recall is the metric that matters — a missed case may go undiagnosed for years, whereas a false alarm costs one unnecessary blood test.

---

## Approach

**1. Data cleaning**
Removed 23,899 exact duplicate rows (9.42% of the dataset) before anything else, so no record could appear in both the training and test sets. Clipped BMI to a physiologically plausible 12–60 range, treating 805 extreme values as data-entry errors rather than deleting the rows.

**2. Target transformation**
Collapsed the three-class target into binary by merging prediabetes and diabetes into one positive class. The prediabetes class was only 1.83% of the data — too small for a classifier to separate reliably. Final positive rate: 17.29%.

**3. Preprocessing**
Standardised the 7 continuous and ordinal columns to zero mean and unit variance; passed the 14 binary indicators through unchanged. The scaler was fitted on the **training split only** — fitting it before the split would leak information about the test set into training and inflate the final score.

**4. Train/test split**
80/20, stratified on the target so both sides preserve the same 17.29% positive rate. Fixed random seed for reproducibility. 183,824 training records, 45,957 held out.

**5. Class imbalance**
Handled at training time rather than by resampling the dataset, using class weighting at a ratio of roughly 4.78:1.

**6. Model comparison**
Four models from four different algorithm families, compared under 5-fold stratified cross-validation within the training set:

- Logistic Regression — the interpretable linear baseline
- Decision Tree — captures non-linear effects a linear model can't
- Gaussian Naive Bayes — the floor against which the others are judged
- LightGBM — gradient-boosted ensemble, the advanced candidate

LightGBM led on six of seven metrics and was selected. Logistic Regression finished within 0.007 ROC-AUC of it and remains the recommended fallback where every decision must be individually explained.

---

## Key Findings

**Survey data alone carries real predictive signal.** ROC-AUC of 0.814 from questions requiring no laboratory test is a genuinely useful result for a screening instrument.

**Five features carry almost all of it.** Permutation importance ranked self-rated general health, BMI, age, high blood pressure and high cholesterol far above the rest. Using only those five retains 98.9% of the full model's ROC-AUC — meaning a screening questionnaire could be cut from 21 questions to about 10 at negligible cost.

**The advanced model's advantage is marginal.** LightGBM beats Logistic Regression by less than one percent of ROC-AUC, which makes the fully interpretable model a serious deployment candidate in its own right.

**The decision threshold is a policy choice, not a modelling one.** At the default 0.50 cut-off the model catches 77.3% of cases at the cost of 11,471 unnecessary tests. Lowering it to 0.33 catches 90.2% but flags nearly half the population. Raising it to 0.62 produces the best F1 score and the highest accuracy — while missing 3,009 diabetic people instead of 1,805. The threshold that looks best on summary statistics is the one that fails worst at the actual task.

**Predictive strength is not causal evidence.** Some of the model's power comes from associations that would not survive being acted upon.

---

## Dataset

- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, via [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Size:** 253,680 respondents, 21 predictor features
- **Features:** medical history (blood pressure, cholesterol, stroke, heart disease), lifestyle (BMI, smoking, physical activity, diet, alcohol), healthcare access, and demographics (age, sex, education, income)
- **Missing values:** none

This dataset was chosen over the alternatives deliberately. Available clinical datasets were too small — the largest held around 770 records. Others were unrealistically clean, suggesting synthetic rather than real data. Several had a 50/50 split between diabetic and non-diabetic cases, which does not reflect reality, since roughly 1 in 12 adults is diagnosed with diabetes.

---

## Repository Structure


00_START_HERE.pdf     Start here — project orientation
01_Report/            Full 32-page technical report with all figures
02_Code/              Analysis and modelling code
03_Dataset/           The BRFSS 2015 data
04_Results/           Figures, confusion matrices, performance charts
05_Environment/       Dependencies and environment setup


---
## Running It

bash
git clone https://github.com/farismqureshi/Diabetes-Prediction-ML-Model.git
cd Diabetes-Prediction-ML-Model
pip install -r 05_Environment/requirements.txt


Then run the notebooks in `02_Code/` in order:

1. 01_data_exploration_and_preprocessing.ipynb` — EDA, cleaning, and the train/test split
2. 02_model_training_and_selection.ipynb` — cross-validated comparison of the four models
3. 03_final_evaluation.ipynb` — final evaluation of LightGBM on the held-out test set

Conda users can use `05_Environment/environment.yml` instead.
## Built With

Python · pandas · NumPy · scikit-learn · LightGBM · Matplotlib · Seaborn

---

## Authors

**Faris Mujtaba Qureshi** 

EE-439 Introduction to Machine Learning, Department of Electrical Engineering, University of Engineering and Technology (UET) Lahore.

---

## References

1. Centers for Disease Control and Prevention. *Behavioral Risk Factor Surveillance System Survey Data, 2015.* U.S. Department of Health and Human Services, Atlanta, Georgia.
2. Teboul, A. *Diabetes Health Indicators Dataset.* Kaggle.
3. Xie, Z., Nikolayeva, O., Luo, J. and Li, D. (2019). Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques. *Preventing Chronic Disease*, 16, E130.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825–2830.

---

*This model is a referral tool, not a diagnostic one. Roughly two in three flagged individuals do not have diabetes. It is appropriate for deciding whom to test first — never as a substitute for an HbA1c or fasting glucose test.*
