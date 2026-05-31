import os

# Create directories
os.makedirs("backend", exist_ok=True)
os.makedirs("frontend", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Write backend/main.py
open("backend/main.py", "w").write('''import io
import os
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from backend.data_processor import PreprocessingPipeline

app = FastAPI(title="Predictive Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = {}

def df_from_upload(file_bytes, filename):
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise HTTPException(400, "Only CSV and Excel files are supported")

def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100)
    return {"mae": round(mae,2), "mse": round(float(mean_squared_error(y_true,y_pred)),2),
            "rmse": round(rmse,2), "r2": round(r2,4), "mape": round(mape,2)}

def _load_html():
    here = os.path.dirname(__file__)
    html_path = os.path.normpath(os.path.join(here, "..", "frontend", "index.html"))
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend not found.</h1>"

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return HTMLResponse(content=_load_html())

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    data = await file.read()
    df = df_from_upload(data, file.filename)
    sid = base64.urlsafe_b64encode(os.urandom(8)).decode()
    store[sid] = {"df": df, "filename": file.filename}
    pipeline = PreprocessingPipeline()
    return {"session_id": sid, "filename": file.filename,
            "rows": len(df), "columns": list(df.columns),
            "profile": pipeline.profile_data(df)}

@app.get("/api/sample/{dataset}")
def load_sample(dataset: str):
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    paths = {"housing": os.path.join(base, "sample_housing.csv"),
             "sales":   os.path.join(base, "sample_sales.csv")}
    if dataset not in paths:
        raise HTTPException(404, "Unknown sample dataset")
    df = pd.read_csv(paths[dataset])
    sid = base64.urlsafe_b64encode(os.urandom(8)).decode()
    store[sid] = {"df": df, "filename": f"sample_{dataset}.csv"}
    pipeline = PreprocessingPipeline()
    return {"session_id": sid, "filename": f"sample_{dataset}.csv",
            "rows": len(df), "columns": list(df.columns),
            "profile": pipeline.profile_data(df)}

class RegressionTrainRequest(BaseModel):
    session_id: str
    target_col: str
    feature_cols: List[str]
    model_type: str = "random_forest"
    fill_numeric: str = "mean"
    fill_categorical: str = "mode"
    scaling: str = "standard"

@app.post("/api/regression/train")
def train_regression(req: RegressionTrainRequest):
    if req.session_id not in store:
        raise HTTPException(404, "Session not found")
    df = store[req.session_id]["df"]
    pipeline = PreprocessingPipeline(fill_numeric=req.fill_numeric,
                                     fill_categorical=req.fill_categorical,
                                     scaling=req.scaling)
    X, y = pipeline.fit_transform(df, req.target_col, req.feature_cols)
    model_map = {
        "linear":            LinearRegression(),
        "ridge":             Ridge(alpha=1.0),
        "random_forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if req.model_type not in model_map:
        raise HTTPException(400, f"Unknown model: {req.model_type}")
    model = model_map[req.model_type]
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    metrics["cv_r2_mean"] = round(float(cv.mean()), 4)
    metrics["cv_r2_std"]  = round(float(cv.std()),  4)
    feature_importance = []
    if hasattr(model, "feature_importances_"):
        fi = sorted(zip(pipeline.all_feature_cols, model.feature_importances_.tolist()), key=lambda x: -x[1])
        feature_importance = [{"feature": f, "importance": round(v,4)} for f,v in fi[:15]]
    elif hasattr(model, "coef_"):
        fi = sorted(zip(pipeline.all_feature_cols, np.abs(model.coef_).tolist()), key=lambda x: -x[1])
        feature_importance = [{"feature": f, "importance": round(v,4)} for f,v in fi[:15]]
    idx = np.random.choice(len(y), min(200, len(y)), replace=False)
    scatter = [{"actual": float(y[i]), "predicted": float(y_pred[i])} for i in idx]
    store[req.session_id]["regression"] = {"model": model, "pipeline": pipeline,
                                            "feature_cols": req.feature_cols,
                                            "target_col": req.target_col}
    return {"metrics": metrics, "feature_importance": feature_importance,
            "scatter": scatter, "model_type": req.model_type,
            "n_features": len(pipeline.all_feature_cols), "n_samples": len(y)}

class RegressionPredictRequest(BaseModel):
    session_id: str
    inputs: dict

@app.post("/api/regression/predict")
def predict_regression(req: RegressionPredictRequest):
    sess = store.get(req.session_id, {}).get("regression")
    if not sess:
        raise HTTPException(404, "No trained model found")
    X_row = sess["pipeline"].transform(req.inputs)
    pred  = sess["model"].predict(X_row)[0]
    return {"prediction": round(float(pred), 2)}

class TimeSeriesTrainRequest(BaseModel):
    session_id: str
    date_col: str
    target_col: str
    model_type: str = "holt_winters"
    forecast_periods: int = 12
    fill_numeric: str = "mean"

@app.post("/api/timeseries/train")
def train_timeseries(req: TimeSeriesTrainRequest):
    if req.session_id not in store:
        raise HTTPException(404, "Session not found")
    df = store[req.session_id]["df"].copy()
    df[req.date_col] = pd.to_datetime(df[req.date_col])
    df = df.sort_values(req.date_col).reset_index(drop=True)
    df[req.target_col] = df[req.target_col].fillna(df[req.target_col].mean())
    y     = df[req.target_col].values.astype(float)
    dates = df[req.date_col]
    freq  = pd.infer_freq(dates) or "MS"
    n_test      = min(12, max(6, int(len(y) * 0.2)))
    y_train     = y[:-n_test]
    y_test      = y[-n_test:]
    dates_train = dates.iloc[:-n_test]
    dates_test  = dates.iloc[-n_test:]
    series_train = pd.Series(y_train, index=pd.DatetimeIndex(dates_train, freq=freq))
    if req.model_type == "holt_winters":
        model     = ExponentialSmoothing(series_train, trend="add", seasonal="add",
                                         seasonal_periods=12, initialization_method="estimated").fit(optimized=True)
        test_pred = model.forecast(n_test)
        future_fc = model.forecast(n_test + req.forecast_periods)[-req.forecast_periods:]
    elif req.model_type == "sarima":
        model     = SARIMAX(series_train, order=(1,1,1), seasonal_order=(1,1,0,12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        test_pred = model.forecast(n_test)
        future_fc = model.forecast(n_test + req.forecast_periods)[-req.forecast_periods:]
    else:
        raise HTTPException(400, f"Unknown model: {req.model_type}")
    mae  = mean_absolute_error(y_test, test_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    mape = float(np.mean(np.abs((y_test - test_pred) / np.where(y_test==0,1e-9,y_test))) * 100)
    future_dates = pd.date_range(dates.iloc[-1], periods=req.forecast_periods+1, freq=freq)[1:]
    return {
        "metrics":          {"mae": round(mae,2), "rmse": round(rmse,2), "mape": round(mape,2)},
        "historical":       [{"date": str(d.date()), "value": float(v)} for d,v in zip(dates, y)],
        "test_predictions": [{"date": str(d.date()), "value": float(v)} for d,v in zip(dates_test, test_pred)],
        "forecast":         [{"date": str(d.date()), "value": round(float(v),2)} for d,v in zip(future_dates, future_fc)],
        "model_type":       req.model_type,
        "n_train":          len(y_train),
        "n_test":           n_test,
    }
''')

print("backend/main.py written:", len(open("backend/main.py").read()), "chars")

# Write backend/__init__.py
open("backend/__init__.py", "w").write("# backend package\n")
print("backend/__init__.py written")

print("All done! Now run: uvicorn backend.main:app --host 127.0.0.1 --port 8000")
