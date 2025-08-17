This is a readme file showingcase the code that was done.

Predicting Median House Prices in Metropolitan Adelaide (2015–2020)

Suburb-level forecasting of median house prices using economic and location-based features with a leakage-safe time-series setup.

Main notebook: Assignment1_PartD_code.ipynb
Outputs: figures in figures/, tables in tables/.

Data

Place CSVs in ./data/ with at least these columns:

Suburb • Year (2015–2020) • Qtr or Quarter (Q1..Q4 or Mar/Jun/Sep/Dec) • MedianPrice

Files can be named freely (e.g., median_prices_2015.csv … median_prices_2019.csv; 2020)

Open Assignment1_PartD_code.ipynb → Kernel → Restart & Run All.
Figures are saved to figures/ and the model comparison table to tables/.

Reproducibility notes

Split: train = 2015–2018, hold-out = 2019 (chronological).

Pooled quarter plot: uses all rows 2015–2019.

Preprocessing: done via Pipeline/ColumnTransformer; fit on train folds only.

Baselines: seasonal-naïve (t-4 with safe fallbacks) and past running mean.

Outputs you should see

tables/table_R1_model_comparison.csv

figures/Fig_R1_Pred_vs_Actual_2019_color.png

figures/Fig_R2_Residuals_vs_Pred_2019_color.png

figures/Fig_R3b_Quarterwise_MAE_Pooled_2015_2019_color.png

figures/Fig_R4_Suburb_MAE_2019_<model>_prettyTicks.png

figures/Fig_M1_Methodology_Flowchart.png



1) LOAD & MERGE RAW FILES
- Reads all CSVs in /data
- Standardises column names (Suburb, Year, Qtr/Quarter, MedianPrice)
- Parses quarter tokens like "Q1" or "Mar/Jun/Sep/Dec"
- Concatenates into a single long panel: one row per (Suburb, Year, Quarter)

2) CLEAN & DATE
- Cast dtypes (Year=int, Qtr=int, MedianPrice=float)
- Drop impossible/empty rows (NaNs for Year/Qtr/MedianPrice)
- Build a PeriodIndex (year+quarter) and a quarter-end timestamp "date"
- Sort by (Suburb, date)

3) FEATURES
- Create lags: y(t-1), y(t-4) per suburb
- Build trend index and quarter dummies (via one-hot later)
- Keep target y = MedianPrice (optionally log-price if toggled)
  
4) Baselines (computed without leakage)
def make_baselines(df):
    """
    Build leakage-safe baselines for each (Suburb, Year, Qtr).
    - pred_snaive: y(t-4) with fallback to y(t-1), then running past-mean up to t-1.
    - pred_mean  : running past-mean up to t-1.
    Returns: df with 'pred_snaive' and 'pred_mean'.

5) SPLIT & PIPELINES
- Train = 2015–2018, Hold-out Test = TEST_YEAR (default 2019)
- ColumnTransformer:
 * numeric: StandardScaler
 * categorical: OneHotEncoder(handle_unknown='ignore') for Suburb/Quarter
- Models tried: Linear, Ridge, Lasso, RandomForest, GradientBoosting
- TimeSeriesSplit + GridSearchCV for Ridge/Lasso (and light grids for tree models)
- Fit on TRAIN only; produce test predictions: pred_ridge, pred_lasso, pred_rf, pred_gb, pred_linear

6) EVALUATION TABLE
- Compute MAE, RMSE, R^2 on hold-out for:
- Seasonal-naïve, Mean baseline, Ridge, RF, GB, Linear, Lasso
- Save to tables/table_R1_model_comparison.csv

Troubleshooting

Quarter/Year parse errors: ensure Year is numeric and Qtr/Quarter is Q1..Q4 or Mar/Jun/Sep/Dec.

“0 sample(s)” error: check that CSVs are in data/ and columns match.

Negative R² for Random Forest: expected on this dataset (poor generalisation); not a bug.


