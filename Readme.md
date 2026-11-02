# Forecasting Hourly Pedestrian Traffic: ARIMA vs UCM vs XGBoost

How many people will walk past a street sensor in the next two months, hour by hour?
This project forecasts **1,439 hours** (1 January – 29 February 2020) of pedestrian counts from a sensor on a public street in Australia, comparing three model families: **ARIMA**, **Unobserved Components Models (UCM)** and **XGBoost**.

> Individual project for the *Streaming Data Management and Time Series Analysis* course — MSc in Data Science, University of Milano-Bicocca (2026). The forecasts were submitted to a class-wide competition ranked by mean absolute error.

---

## The data

- **41,328** hourly observations (15 April 2015 – 31 December 2019), counts from 0 to 906
- **Right-skewed** distribution with high variability (coefficient of variation 102%)
- **Multiple seasonality**: a daily cycle peaking at 2–4 PM, a weekly cycle (weekends −28% vs weekdays) and an annual cycle (peak in the Australian summer, January–February)
- **Strong holiday effects**: Christmas −88%, New Year's Day −75%, Australia Day −63%

---

## Approach

All three models share the same preprocessing: a **Box-Cox transformation** (λ = 0.249, Guerrero's method, offset 0.5 for zero counts) to stabilise the variance, and **holiday dummies** for the calendar effects.

| Model | Specification |
|---|---|
| **ARIMA** | Regression with ARIMA(5,0,3) errors. Daily, weekly and annual seasonality modelled with **Fourier terms**, with the number of harmonics K = (12, 6, 1) chosen by AICc grid search. **d = 0**: with differencing, the forecast stayed anchored to the low level of late December (~17 pedestrians/hour instead of ~80). |
| **UCM** | Three steps: OLS on 47 deterministic regressors (Fourier terms, day-of-week and holiday dummies, R² = 0.917), then a **Local Level model** on the residuals estimated with the Kalman filter (`KFAS`), then recombination of the two components. Adopted because stochastic trigonometric seasonals in KFAS did not converge. |
| **XGBoost** | Time series reframed as supervised regression: cyclical calendar features (sine/cosine), **lags** (24, 48, 168, 336 hours), rolling means and standard deviations. Hyperparameters tuned with a 3-fold **walk-forward cross-validation**. The lag of one week (`lag_168`) alone accounts for 62% of the gain. |

---

## Results

| | ARIMA | UCM | XGBoost |
|---|---|---|---|
| In-sample MAE (Box-Cox scale) | 0.877 | 0.813 | **0.476** |
| In-sample RMSE (Box-Cox scale) | 1.142 | 1.093 | **0.626** |
| Mean forecast / historical Jan–Feb mean | **0.97** | **0.98** | ~0.94 |
| Residual autocorrelation | spikes at 24h | spikes at 24h | **nearly flat** |

**Main takeaway:** XGBoost fits the past best, but **ARIMA and UCM give the more reliable forecast**. Beyond a one-week horizon, XGBoost's lag features can only rely on December values — a seasonal minimum — so it underestimates the summer level. A better in-sample fit does not mean a better forecast.

<!-- Add a figure here, e.g. the three forecasts compared:
<p align="center"><img src="images/forecast_comparison.png" width="85%" alt="ARIMA, UCM and XGBoost forecasts"></p>
-->

---

## Limitations

- **Lag bias in XGBoost** for long horizons (see above); a recursive strategy would reduce it but propagate errors.
- **In-sample comparison**: ARIMA and UCM were not re-estimated on cross-validation folds (too costly on ~41,000 observations).
- **Residual autocorrelation at 24h** in ARIMA and UCM: part of the daily structure is not fully captured.
- **No prediction intervals for XGBoost**; possible extension with quantile regression.

---

## Repository structure

```
├── scripts/
│   └── 01_arima.R                   # ARIMA with Fourier regressors and holiday dummies
├── forecasts/
│   └── forecasts_jan_feb_2020.csv   # submitted forecasts (ARIMA, UCM, ML)
├── report/
│   └── report.pdf                   # full methodology, diagnostics and figures
└── README.md
```

The full methodology for all three models — specifications, parameter estimates, diagnostics and figures — is documented in the [report](report/report.pdf). The UCM and XGBoost scripts will be added to this repository.

**How to run:** place `student_dataset.csv` (course material, not included) in the working directory and run `scripts/01_arima.R` in R. Estimation takes about 20 minutes on a standard laptop.

---

## Tools

R · forecast · KFAS · xgboost · tseries · ggplot2

---

## Author

**Carlo Pegoraro** — [GitHub](https://github.com/carlopego14)
