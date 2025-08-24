"""
Streamlit manual-only tester — loads a local model (`model.pkl`) and builds a manual input form from either:
 - `model_meta.json` (recommended), or
 - the model's `feature_names_in_` if present, or
 - a pasted comma-separated header, or
 - an uploaded sample CSV.

This version **does NOT** auto-load any `test-data.csv`. You will enter the test values manually using the form.

Run: `streamlit run streamlit_manual_only.py`
"""

import streamlit as st
import joblib
import pickle
import pandas as pd
import os
import tempfile
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Manual model tester (manual entry)", layout="centered")
st.title("Manual input — enter one row")

# -------------------- Helpers --------------------

def load_model_from_path(path: str):
    if not path:
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None


def infer_schema_from_df(df: pd.DataFrame):
    cols = list(df.columns)
    dtypes = {}
    categories = {}
    defaults = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            dtypes[c] = 'numeric'
            med = df[c].median()
            defaults[c] = int(med) if pd.notna(med) and float(med).is_integer() else (float(med) if pd.notna(med) else 0)
        else:
            nunique = df[c].nunique(dropna=True)
            if nunique <= 40:
                cats = sorted(df[c].dropna().unique().tolist())
                categories[c] = cats
                try:
                    defaults[c] = df[c].mode().iloc[0]
                except Exception:
                    defaults[c] = ''
            else:
                defaults[c] = ''
    return cols, dtypes, categories, defaults

# -------------------- Load model --------------------
DEFAULT_MODEL = "model.pkl"
model = None
model_path = DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else None
if model_path:
    model = load_model_from_path(model_path)
    if model is not None:
        st.success(f"Auto-loaded model from {model_path}")
    else:
        st.warning(f"Found {model_path} but failed to load it. You can upload a model file below.")
else:
    st.info(f"No local '{DEFAULT_MODEL}' found — if your model is elsewhere you can upload it.")

# fallback uploader
if model is None:
    uploaded_model = st.file_uploader("Upload model file (joblib / pkl)", type=["joblib", "pkl", "sav", "gzip", "bz2", "zip"]) 
    if uploaded_model is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_model.name)[1])
        tmp.write(uploaded_model.getbuffer())
        tmp.flush(); tmp.close()
        model = load_model_from_path(tmp.name)
        if model is not None:
            st.success("Uploaded model loaded")

if model is None:
    st.error("No model available. Place 'model.pkl' in this folder or upload one to continue.")
    st.stop()

# -------------------- Discover schema (NO auto-loading of test CSV) --------------------
cols = None
dtypes = {}
categories = {}
defaults = {}

# 1) Optional: upload model_meta.json
uploaded_meta = st.file_uploader("Optional: upload model_meta.json (columns/dtypes/categories)", type=["json"])
if uploaded_meta is not None:
    try:
        meta_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        meta_path.write(uploaded_meta.getbuffer())
        meta_path.flush(); meta_path.close()
        with open(meta_path.name, 'r') as f:
            meta = json.load(f)
        if isinstance(meta, dict) and meta.get('columns'):
            cols = list(meta.get('columns'))
            dtypes = meta.get('dtypes', {}) or {}
            categories = meta.get('categories', {}) or {}
            defaults = meta.get('defaults', {}) or {}
            st.success('Using schema from uploaded model_meta.json')
    except Exception as e:
        st.error(f'Failed to read model_meta.json: {e}')

# 2) Try model.feature_names_in_ (if available)
if cols is None:
    try:
        if hasattr(model, 'feature_names_in_'):
            cols = list(model.feature_names_in_)
            st.success('Using column names from model.feature_names_in_')
    except Exception:
        pass

# 3) Ask user to paste header or upload a small CSV to infer columns
if cols is None:
    st.info('No schema found. Please either paste a comma-separated header row, or upload a small CSV with the header that matches the model input columns.')
    pasted = st.text_input('Or paste comma-separated header row (e.g. Year,Mileage,Location)')
    uploaded_sample = st.file_uploader('Optional: upload sample CSV to infer columns', type=['csv'], key='sample_manual')

    if pasted and not cols:
        cols = [c.strip() for c in pasted.split(',') if c.strip()]
        defaults = {c: '' for c in cols}
        st.success('Using pasted header')

    if uploaded_sample is not None and cols is None:
        try:
            df_sample = pd.read_csv(uploaded_sample)
            cols, dtypes, categories, defaults = infer_schema_from_df(df_sample)
            st.success('Inferred schema from uploaded sample CSV')
        except Exception as e:
            st.error(f'Failed to read uploaded CSV: {e}')

if cols is None:
    st.error('No input columns discovered — provide model_meta.json, paste a header, or upload a small CSV.')
    st.stop()

st.write(f"Detected columns: {cols}")

# -------------------- Manual form (one row) --------------------
st.markdown("---")
st.write("Fill the fields (values should match types used during training) and click **Predict**")
with st.form('manual_form'):
    user_inputs = {}
    for c in cols:
        if c in categories and isinstance(categories[c], (list, tuple)) and len(categories[c]) > 0:
            try:
                user_inputs[c] = st.selectbox(c, options=categories[c], index=0)
            except Exception:
                user_inputs[c] = st.selectbox(c, options=categories[c])
        else:
            if dtypes.get(c, '') == 'numeric':
                default = defaults.get(c, 0)
                try:
                    if isinstance(default, (int,)):
                        user_inputs[c] = st.number_input(c, value=int(default))
                    else:
                        user_inputs[c] = st.number_input(c, value=float(default))
                except Exception:
                    user_inputs[c] = st.number_input(c, value=0.0)
            else:
                user_inputs[c] = st.text_input(c, value=str(defaults.get(c, '')))

    submit = st.form_submit_button('Predict')

if submit:
    try:
        row = pd.DataFrame([user_inputs], columns=cols)
        for c in cols:
            if dtypes.get(c) == 'numeric':
                row[c] = pd.to_numeric(row[c], errors='coerce')
        preds = model.predict(row)
        try:
            pred_val = preds[0]
            if hasattr(pred_val, '__len__') and not isinstance(pred_val, (str, bytes)):
                st.write('Predicted (first row, multi-output):')
                st.write(list(pred_val))
            else:
                st.metric('Predicted value', value=round(float(pred_val), 6))
        except Exception:
            st.write('Prediction:')
            st.write(preds)
    except Exception as e:
        st.error(f'Prediction failed: {e}')
