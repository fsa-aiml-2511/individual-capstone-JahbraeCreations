"""
Streamlit capstone app — used vehicle listing price (regression) + price tier (classification).
Run from project root:  streamlit run app/app.py
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=2)
def _load_bundle(filename: str) -> dict:
    path = ROOT / "models" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run notebooks 02 and 03 (after 01) so the pickles are saved under models/."
        )
    return joblib.load(path)


def load_regression_bundle() -> dict:
    return _load_bundle("regression_model.pkl")


def load_classification_bundle() -> dict:
    return _load_bundle("classification_model.pkl")


def inputs_to_frame(
    year: int,
    odometer: float,
    manufacturer: str,
    fuel: str,
    transmission: str,
    drive: str,
    vehicle_type: str,
) -> pd.DataFrame:
    reg = load_regression_bundle()
    age_base = int(reg["age_base"])
    vehicle_age = age_base - int(year)
    row = {
        "vehicle_age": vehicle_age,
        "odometer": float(odometer),
        "manufacturer": manufacturer,
        "fuel": fuel,
        "transmission": transmission,
        "drive": drive,
        "type": vehicle_type,
    }
    cols = reg["feature_cols"]
    return pd.DataFrame([row])[cols]


def predict_price(
    year: int,
    odometer: float,
    manufacturer: str,
    fuel: str,
    transmission: str,
    drive: str,
    vehicle_type: str,
) -> float:
    reg = load_regression_bundle()
    x = inputs_to_frame(year, odometer, manufacturer, fuel, transmission, drive, vehicle_type)
    return float(reg["pipeline"].predict(x)[0])


def predict_tier(
    year: int,
    odometer: float,
    manufacturer: str,
    fuel: str,
    transmission: str,
    drive: str,
    vehicle_type: str,
) -> str:
    cls = load_classification_bundle()
    x = inputs_to_frame(year, odometer, manufacturer, fuel, transmission, drive, vehicle_type)
    return str(cls["pipeline"].predict(x)[0])


def tier_legend() -> str:
    reg = load_regression_bundle()
    q1, q2 = reg["tier_cutoffs"]
    return (
        f"**Budget** = predicted listing price roughly under **${q1:,.0f}** · "
        f"**Mid** = about **${q1:,.0f} – ${q2:,.0f}** · "
        f"**Premium** = roughly above **${q2:,.0f}** (based on tertiles from the cleaned training sample)."
    )


def dropdown_options():
    reg = load_regression_bundle()
    levels = reg["category_levels"]
    return levels["manufacturer"], levels["fuel"], levels["transmission"], levels["drive"], levels["type"]


st.set_page_config(page_title="Used Vehicle Price Predictor", layout="centered")

st.title("Used vehicle price predictor")
st.markdown(
    "This is my capstone mini-app for Craigslist-style vehicle listings. "
    "You punch in a few fields that actually mattered in my models, and you get a **dollar estimate** plus a **budget / mid / premium** bucket."
)

with st.sidebar:
    st.subheader("What the tiers mean")
    st.markdown(tier_legend())
    st.caption("Bins are tertiles from my cleaned sample — same logic as notebook 03.")

try:
    mfg, fuels, trans, drives, types = dropdown_options()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

year_min, year_max = 1990, 2022

tab_reg, tab_cls = st.tabs(["Regression — price ($)", "Classification — price tier"])

with tab_reg:
    st.markdown("### Inputs (what my final regression actually uses)")
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Model year", min_value=year_min, max_value=year_max, value=2015, step=1)
        odometer = st.number_input("Odometer (miles)", min_value=0.0, max_value=400_000.0, value=85_000.0, step=500.0)
        manufacturer = st.selectbox("Manufacturer", mfg)
        fuel = st.selectbox("Fuel type", fuels)
    with c2:
        transmission = st.selectbox("Transmission", trans)
        drive = st.selectbox("Drive", drives)
        vehicle_type = st.selectbox("Vehicle type", types)

    if st.button("Predict price", type="primary", key="reg_btn"):
        try:
            price = predict_price(year, odometer, manufacturer, fuel, transmission, drive, vehicle_type)
            st.success(f"Estimated listing price: **${price:,.0f}**")
            st.caption("Not financial advice — just what the model learned from messy real-world listings.")
        except Exception as ex:  # noqa: BLE001
            st.error(f"Something broke while predicting: {ex}")

with tab_cls:
    st.markdown("### Same inputs → tier instead of dollars")
    c1, c2 = st.columns(2)
    with c1:
        year_c = st.number_input(
            "Model year (classification tab)", min_value=year_min, max_value=year_max, value=2015, step=1, key="y2"
        )
        odometer_c = st.number_input(
            "Odometer miles (classification tab)", min_value=0.0, max_value=400_000.0, value=85_000.0, step=500.0, key="o2"
        )
        manufacturer_c = st.selectbox("Manufacturer (classification)", mfg, key="m2")
        fuel_c = st.selectbox("Fuel type (classification)", fuels, key="f2")
    with c2:
        transmission_c = st.selectbox("Transmission (classification)", trans, key="t2")
        drive_c = st.selectbox("Drive (classification)", drives, key="d2")
        vehicle_type_c = st.selectbox("Vehicle type (classification)", types, key="v2")

    if st.button("Predict tier", type="primary", key="cls_btn"):
        try:
            tier = predict_tier(
                year_c, odometer_c, manufacturer_c, fuel_c, transmission_c, drive_c, vehicle_type_c
            )
            st.success(f"Predicted tier: **{tier}**")
        except Exception as ex:  # noqa: BLE001
            st.error(f"Something broke while predicting: {ex}")
