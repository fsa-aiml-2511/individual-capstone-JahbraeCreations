# Individual capstone — used vehicle listing price

**Author:** Anthony Howell  
**Dataset:** Craigslist-style used vehicle listings (`vehicles.csv`)  
**Goal:** Predict **listing price** (regression) and **price tier** — budget / mid / premium — from tertiles (classification).

## Repository layout

```text
NEW/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              ← optional: place full vehicles.csv here (see data/raw/README.txt)
│   └── processed/        ← cleaned_data.csv produced by notebook 01
├── models/
│   ├── regression_model.pkl      ← notebook 02
│   └── classification_model.pkl  ← notebook 03
├── notebooks/
│   ├── 01_problem_statement_and_eda.ipynb
│   ├── 02_regression_model.ipynb
│   └── 03_classification_model.ipynb
└── app/
    └── app.py            ← Streamlit UI
```

## Setup

```bash
cd NEW
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing data and models

1. Ensure your full **`vehicles.csv`** is available. Notebook 01 may use an absolute `RAW_PATH`; edit that cell if you move the file or copy it to `data/raw/`.
2. Run notebooks **in order**: `01` → `02` → `03`.
3. Confirm outputs exist:
   - `data/processed/cleaned_data.csv`
   - `models/regression_model.pkl`
   - `models/classification_model.pkl`

## Run the Streamlit app

From the **`NEW`** directory (project root):

```bash
streamlit run app/app.py
```

The app loads both pickles from `models/` and uses the same seven features as the notebooks: vehicle age, odometer, manufacturer, fuel, transmission, drive, and type.

## Notes

- **Tier labels** match notebook 03: tertiles of cleaned `price` → budget / mid / premium.
- **Model year** inputs are limited to **1990–2022** to match the cleaning rules in notebook 01.
- If `regression_model.pkl` is missing, the app shows a clear error until you run notebook 02.
