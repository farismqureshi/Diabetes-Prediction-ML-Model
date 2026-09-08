"""
Inference API for the EE-439 diabetes risk model.

Put these five files in one folder:

    app.py                        this file
    index.html                    the front end
    final_model_lightgbm.joblib   from 04_Results/models/ in the project Drive
    brfss_preprocessor.joblib     from 04_Results/models/ in the project Drive

Then:

    pip install fastapi uvicorn lightgbm scikit-learn pandas numpy joblib shap
    uvicorn app:app --reload --port 8000

and open http://127.0.0.1:8000/ — this server serves the page as well as the
model, so page and API share an origin and no CORS or local-network permission
gets in the way. http://127.0.0.1:8000/health reports what loaded.

WHY THE PREPROCESSOR MATTERS
----------------------------
Notebook 01 fitted a ColumnTransformer that standardises seven columns
(BMI, MentHlth, PhysHlth, GenHlth, Age, Education, Income) and passes the
fourteen binary flags through untouched. The model was trained on that
transformed matrix, never on raw survey answers. Feeding it a raw record
produces confident nonsense — BMI 31 instead of +0.9 standard deviations is
simply a different question. So every request goes through the same fitted
transformer before it reaches the model.

The transformer also reorders the columns: it emits the seven scaled columns
first, then the fourteen binary ones. That reordering is handled inside
`preprocessor.transform`, so this file only has to supply the RAW column order
below — the order the CSV had in notebook 01.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODEL_PATH = HERE / "final_model_lightgbm.joblib"
PREPROCESSOR_PATH = HERE / "brfss_preprocessor.joblib"
PAGE_PATH = HERE / "index.html"

# Raw column order, exactly as X looked in notebook 01 before transforming.
RAW_ORDER: List[str] = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# What the transformer emits, in order: scaled columns first, then passthrough.
SCALE_COLS = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]
BINARY_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
TRANSFORMED_ORDER = SCALE_COLS + BINARY_COLS

# Notebook 01 clipped BMI into a clinical range before fitting the scaler.
BMI_MIN, BMI_MAX = 12.0, 60.0

VALUE_LABELS = {
    "GenHlth": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "Sex": {0: "Female", 1: "Male"},
    "Age": {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
        7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
        12: "75-79", 13: "80+",
    },
}
UNITS = {"BMI": "kg/m²", "MentHlth": "days", "PhysHlth": "days"}


# --------------------------------------------------------------------------
# Request / response schema
# --------------------------------------------------------------------------
class Respondent(BaseModel):
    HighBP: int = Field(ge=0, le=1)
    HighChol: int = Field(ge=0, le=1)
    CholCheck: int = Field(ge=0, le=1)
    BMI: float = Field(ge=10, le=100)
    Smoker: int = Field(ge=0, le=1)
    Stroke: int = Field(ge=0, le=1)
    HeartDiseaseorAttack: int = Field(ge=0, le=1)
    PhysActivity: int = Field(ge=0, le=1)
    Fruits: int = Field(ge=0, le=1)
    Veggies: int = Field(ge=0, le=1)
    HvyAlcoholConsump: int = Field(ge=0, le=1)
    GenHlth: int = Field(ge=1, le=5)
    MentHlth: int = Field(ge=0, le=30)
    PhysHlth: int = Field(ge=0, le=30)
    DiffWalk: int = Field(ge=0, le=1)
    AnyHealthcare: int = Field(ge=0, le=1)
    NoDocbcCost: int = Field(ge=0, le=1)
    Sex: int = Field(ge=0, le=1)
    Age: int = Field(ge=1, le=13)
    Education: int = Field(ge=1, le=6)
    Income: int = Field(ge=1, le=8)


class Factor(BaseModel):
    feature: str
    value: str
    contribution: float


class Prediction(BaseModel):
    probability: float
    model: str
    factors: List[Factor] = []


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------
app = FastAPI(title="Diabetes Risk Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # coursework demo; narrow this for anything real
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

model = None
preprocessor = None
explainer = None
model_name = "unloaded"
transform_order = RAW_ORDER       # replaced at startup by the transformer's own order


@app.on_event("startup")
def load_artifacts() -> None:
    global model, preprocessor, explainer, model_name, transform_order

    import joblib

    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        # Trust the transformer's own record of the columns it was fitted on,
        # rather than assuming RAW_ORDER is right.
        names = getattr(preprocessor, "feature_names_in_", None)
        if names is not None:
            transform_order = list(names)
            if set(transform_order) != set(RAW_ORDER):
                print("[warn] preprocessor columns differ from RAW_ORDER:",
                      set(transform_order) ^ set(RAW_ORDER))
        print(f"[ok] loaded preprocessor ({len(transform_order)} input columns)")
    else:
        print(f"[error] {PREPROCESSOR_PATH.name} not found — /predict will return 503.")
        return

    if not MODEL_PATH.exists():
        print(f"[error] {MODEL_PATH.name} not found — /predict will return 503.")
        return

    model = joblib.load(MODEL_PATH)
    model_name = type(model).__name__
    print(f"[ok] loaded {model_name} from {MODEL_PATH.name}")

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        print("[ok] SHAP explainer ready")
    except Exception as exc:  # explanations are optional
        print(f"[warn] SHAP unavailable, factors will be empty: {exc}")


def pretty_value(feature: str, raw: float) -> str:
    if feature in BINARY_COLS:
        return "Yes" if raw >= 0.5 else "No"
    if feature in VALUE_LABELS:
        return VALUE_LABELS[feature].get(int(raw), str(int(raw)))
    unit = UNITS.get(feature)
    return f"{raw:g} {unit}" if unit else f"{raw:g}"


def top_factors(matrix: np.ndarray, raw: dict, k: int = 6) -> List[Factor]:
    """SHAP values live in TRANSFORMED column order; labels use raw values."""
    if explainer is None:
        return []

    values = explainer.shap_values(matrix)
    if isinstance(values, list):          # older SHAP returns one array per class
        values = values[1]
    values = np.asarray(values).reshape(-1)

    if len(values) != len(TRANSFORMED_ORDER):
        return []

    order = np.argsort(-np.abs(values))[:k]
    return [
        Factor(
            feature=TRANSFORMED_ORDER[i],
            value=pretty_value(TRANSFORMED_ORDER[i], float(raw[TRANSFORMED_ORDER[i]])),
            contribution=float(values[i]),
        )
        for i in order
    ]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def page():
    """Serve the front end from the same origin as the API."""
    if not PAGE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"index.html not found. Put it next to app.py at {PAGE_PATH}.",
        )
    return FileResponse(PAGE_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if (model is not None and preprocessor is not None) else "not ready",
        "model": model_name,
        "model_file": MODEL_PATH.name if MODEL_PATH.exists() else "MISSING",
        "preprocessor_file": PREPROCESSOR_PATH.name if PREPROCESSOR_PATH.exists() else "MISSING",
    }


@app.post("/predict", response_model=Prediction)
def predict(respondent: Respondent) -> Prediction:
    if model is None or preprocessor is None:
        missing = [
            p.name for p in (MODEL_PATH, PREPROCESSOR_PATH) if not p.exists()
        ]
        raise HTTPException(
            status_code=503,
            detail=f"Not ready. Missing next to app.py: {', '.join(missing) or 'nothing — check the startup log'}.",
        )

    raw = respondent.model_dump()
    # Match notebook 01's clipping so the scaler sees the range it was fitted on.
    raw["BMI"] = float(np.clip(raw["BMI"], BMI_MIN, BMI_MAX))

    frame = pd.DataFrame([[raw[c] for c in transform_order]], columns=transform_order)
    matrix = preprocessor.transform(frame)

    probability = float(model.predict_proba(matrix)[0][1])

    return Prediction(
        probability=probability,
        model=model_name,
        factors=top_factors(matrix, raw),
    )
