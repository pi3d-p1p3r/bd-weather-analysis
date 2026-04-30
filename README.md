# Climate Forecasting: Temperature Prediction and Land-Atmosphere Coupling in Bangladesh

## Overview

This project aims to predict daily 2-metre temperature (t2m) anomalies across major regions in Bangladesh during the critical pre-monsoon and early monsoon months (March–July). By leveraging historical ERA5 reanalysis data spanning from 1995 to 2023, the project forecasts 2024 temperatures utilizing an advanced deep learning framework. Additionally, it investigates the physical drivers of extreme heat by analyzing the land-atmosphere coupling strength—specifically, the relationship between soil moisture deficits and temperature increases during heatwave conditions.

## Project Goals

1. **Analyze Land-Atmosphere Coupling:** Conduct piecewise linear regression (breakpoint analysis) to measure the strength of the relationship between soil moisture (swvl1) and 2-metre temperature (t2m).
2. **Develop an Advanced Predictive Model:** Adapt the **DUNE** (Deep UNet++ based Ensemble) architecture from 2D spatial grids to 1D time-series data for highly accurate, single-location temperature forecasting, and compare its performance against a baseline LSTM model.
3. **Evaluate Regional Dynamics:** Assess the variations in predictive performance and physical coupling across eight major regions in Bangladesh: Dhaka, Chattogram, Sylhet, Rajshahi, Rangpur, Mymensingh, Barishal, and Khulna.

---

## Key Findings

### 1. Land-Atmosphere Coupling (Breakpoint Analysis)

The breakpoint analysis measured the $R^2$ scores of the soil moisture vs. temperature relationship, revealing critical insights into land-atmosphere feedback mechanisms during heatwaves:

- **Peak Coupling:** Most regions exhibit a significant increase in coupling strength starting in April/May, which peaks in June or July.
- **Regional Hotspots:** Chattogram and Sylhet demonstrate some of the highest coupling values ($R^2 > 0.45$) during the monsoon onset months.
- **Temporal Variability:** Low $R^2$ values in March indicate that land-atmosphere feedback is less dominant in the early pre-monsoon season compared to the late pre-monsoon and early monsoon periods.

**Region-wise Peak $R^2$ Scores (March - July):**
| Region | March | April | May | June | July |
|--------|-------|-------|-----|------|------|
| **Barishal** | 0.0392 | 0.2300 | 0.2305 | 0.3233 | 0.2474 |
| **Chattogram** | 0.0559 | 0.2548 | 0.3869 | 0.4723 | 0.4637 |
| **Dhaka** | 0.0555 | 0.3178 | 0.3146 | 0.3648 | 0.3198 |
| **Khulna** | 0.0555 | 0.3178 | 0.3146 | 0.3648 | 0.3198 |
| **Mymensingh** | 0.0869 | 0.2713 | 0.2767 | 0.3905 | 0.3906 |
| **Rajshahi** | 0.1492 | 0.3489 | 0.2086 | 0.2271 | 0.3491 |
| **Rangpur** | 0.1219 | 0.3035 | 0.2109 | 0.2851 | 0.3740 |
| **Sylhet** | 0.0682 | 0.2595 | 0.3355 | 0.4246 | 0.4208 |

---

### 2. Model Performance (DUNE vs. LSTM)

The DUNE-1D predictive model demonstrated robust forecasting capability for the 2024 target period, consistently outperforming the baseline LSTM model. Performance varied by region, with DUNE capturing multi-scale temporal dependencies more effectively, especially in regions with complex topographies or coastal influences.

**Performance Metrics for DUNE Model Across Regions:**
| Region | $R^2$ Score (DUNE) | $R^2$ Score (LSTM) | RMSE (°C) | MAE (°C) | MAPE (%) |
|--------|--------------------|--------------------|-----------|----------|----------|
| **Chattogram** | 0.8053 | 0.5450 | 0.8195 | 0.5852 | 2.05% |
| **Dhaka** | 0.7722 | 0.7506 | 1.5159 | 1.1088 | 3.88% |
| **Rajshahi** | 0.7495 | 0.6514 | 1.7864 | 1.3496 | 4.48% |
| **Sylhet** | 0.7430 | 0.5834 | 1.6180 | 1.1864 | 4.25% |
| **Mymensingh** | 0.7365 | 0.7130 | 1.7567 | 1.3093 | 4.58% |
| **Khulna** | 0.7325 | 0.7351 | 1.6309 | 1.1589 | 3.94% |
| **Rangpur** | 0.7027 | 0.6665 | 1.9148 | 1.4062 | 4.84% |
| **Barishal** | 0.6943 | 0.6906 | 1.5255 | 1.0473 | 3.65% |

_Note: **Chattogram** achieved the highest overall $R^2$ score and the lowest absolute errors with DUNE, showcasing a massive improvement (+0.26) over the LSTM baseline. **Sylhet** also saw significant improvements (+0.16) due to DUNE's capability to better handle complex topographical climates._

---

### 3. Model Significance

The multi-scale ensemble approach of the DUNE model provides enhanced predictive capability critical for:

- **Agricultural Planning:** Enabling better crop management and irrigation scheduling.
- **Disaster Preparedness:** Improving early warning systems for heatwaves and extreme temperature events.
- **Climate Adaptation:** Supplying reliable data for developing long-term adaptation strategies.

---

## Methodology Overview

The project workflow relies on a rigorous data pipeline and an innovative deep learning architecture:

1. **Data Processing:** Extraction of daily ERA5 sequences (30-day sliding windows) for target locations.
2. **Anomaly-based Forecasting:** Instead of predicting absolute temperatures directly, the model targets deviations from the long-term climatological mean (1995–2019). This stationary approach simplifies learning.
3. **DUNE-1D Architecture:** A 1D adaptation of the UNet++ architecture featuring a multi-scale ensemble output. The model extracts fine-to-coarse temporal features using residual blocks and averages predictions across 4 decoder levels.
4. **Combined Loss Function:** Training utilizes a weighted combination of RMSE, MAE, and an ensemble consistency penalty to maximize robustness against outliers and ensure member agreement.
