# Diabetes Risk Screening — Web Interface

A browser front end for the LightGBM diabetes risk model trained on the CDC's
BRFSS 2015 health indicators survey. Twenty-one survey answers in, one screening
probability out.

Built as the deliverable interface for **EE-439 Introduction to Machine
Learning**, University of Engineering and Technology, Lahore.

![The screening interface, scoring an example respondent](screenshots/interface.png)

<p align="center"><em>Twenty-one BRFSS indicators on the left; probability, threshold and SHAP contributors on the right.</em></p>

### The result panel

![Predicted risk, threshold presets, and leading contributors](screenshots/result-panel.png)

The probability is banded against an adjustable decision threshold and compared
against the 17.3% cohort prevalence. Contributing factors come from SHAP values
computed per request, so the panel explains *this* prediction rather than the
model in general.

> **Running it locally is required for live predictions.** The interface needs
> the Python API in this repo running alongside it — see
> [Running it](#running-it). There is no hosted demo.

---

## What it does

The page collects the twenty-one BRFSS indicators the model was trained on,
grouped as the survey groups them — clinical history, lifestyle, general health
and function, access to care, demographics. Each control shows the encoded value
it will send, so the request can be checked against the model's expectations at
a glance.

The result panel reports:

- the predicted probability, banded against an adjustable decision threshold
- where that probability sits relative to the 17.3% cohort prevalence
- the six features contributing most to this particular prediction, from SHAP
  values, signed by direction of effect
- one-click presets for the three operating points identified in the project's
  final evaluation notebook

## Model

| | |
|---|---|
| Algorithm | LightGBM (`n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`) |
| Class imbalance | `scale_pos_weight ≈ 4.78` |
| Training set | 183,824 records |
| Held-out test set | 45,957 records |
| Test ROC-AUC | 0.8142 |
| Test PR-AUC | 0.4712 |
| Recall at threshold 0.50 | 0.7728 |
| Precision at threshold 0.50 | 0.3486 |

The positive class is prediabetes **or** diabetes, following Kaggle's
`diabetes_binary` definition.

### Operating points

| Threshold | Basis | Effect |
|---|---|---|
| 0.33 | Recall 0.90 on out-of-fold predictions | Catches 90% of cases, flags nearly half the population |
| 0.50 | Default, recommended | The reported operating point |
| 0.62 | Best F1 | Higher precision, considerably more missed cases |

Because the model was trained with `scale_pos_weight`, its output probabilities
are deliberately inflated relative to real-world prevalence. A 0.60 output means
"above the 0.50 cut-off", not "a 60% chance of diabetes".

## Running it

The two model artefacts are produced by the project notebooks and must sit
beside `app.py`:

```
app.py
index.html
final_model_lightgbm.joblib     # from notebook 03
brfss_preprocessor.joblib       # from notebook 01
```

Then:

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

and open <http://127.0.0.1:8000/>. The server serves the page as well as the
model, so both share an origin and no CORS configuration is needed. Opening
`index.html` directly off disk will not work — browsers block a `file://` page
from calling `127.0.0.1`.

`GET /health` reports which artefacts loaded.

## Why the preprocessor is not optional

The model was never trained on raw survey answers. Notebook 01 fits a
`ColumnTransformer` that standardises seven columns — BMI, MentHlth, PhysHlth,
GenHlth, Age, Education, Income — and passes the fourteen binary flags through
unchanged. `app.py` loads that fitted transformer and applies it to every
request before the model sees it. Skipping this step does not raise an error; it
silently produces confident nonsense, because BMI 31 and BMI at +0.9 standard
deviations are different questions.

The transformer also reorders columns (scaled first, then passthrough), which is
why `app.py` supplies the raw CSV order and lets `transform` handle the rest.

## API

`POST /predict` — all twenty-one features as a flat JSON object:

```json
{
  "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 31,
  "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
  "PhysActivity": 0, "Fruits": 0, "Veggies": 1, "HvyAlcoholConsump": 0,
  "GenHlth": 3, "MentHlth": 4, "PhysHlth": 8, "DiffWalk": 0,
  "AnyHealthcare": 1, "NoDocbcCost": 0,
  "Sex": 1, "Age": 9, "Education": 4, "Income": 5
}
```

Response:

```json
{
  "probability": 0.6117,
  "model": "LGBMClassifier",
  "factors": [
    { "feature": "GenHlth", "value": "Good", "contribution": 0.81 }
  ]
}
```

Encodings follow the BRFSS 2015 codebook: `GenHlth` 1–5 (excellent to poor),
`Age` 1–13 (18–24 through 80+), `Education` 1–6, `Income` 1–8, everything else
binary except `BMI`, `MentHlth` and `PhysHlth`. BMI is clipped to [12, 60] to
match the range the scaler was fitted on.

## Limitations

Precision sits near 0.35 — roughly two in three flagged people do not have
diabetes. For a screening instrument whose false alarms lead to an inexpensive
blood test that is a defensible trade, but the output is a referral list, not a
diagnosis, and no substitute for HbA1c or fasting glucose testing.

Every input is self-reported survey data, and the BRFSS target records whether a
respondent was ever *told* they have diabetes — so the model predicts diagnosis,
not disease, and inherits whatever access-to-care bias sits behind who gets
diagnosed in the first place.

## Project

Full pipeline — data exploration and preprocessing, model training and
selection, final evaluation — lives in the project's notebooks. Model selection
compared Logistic Regression, Decision Tree, Gaussian Naive Bayes and LightGBM
by 5-fold stratified cross-validation; LightGBM won on ROC-AUC by 0.0067 over
Logistic Regression.
