# Bike Sharing Demand Prediction

## Project Overview

This project focuses on predicting hourly bike rental demand using machine learning. It uses historical bike rental data along with weather and time-related features to estimate the number of bikes rented during a given hour.

The goal is to build, evaluate, and deploy a machine learning model that can predict hourly rental demand. The project covers the main stages of a machine learning workflow: exploratory data analysis, data preparation, model training and comparison, evaluation, and deployment through an API and Docker.

The project uses the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

Accurate demand forecasts can help bike-sharing operators plan bicycle availability, allocate resources more efficiently, and prepare for changes in demand throughout the day.

**Final model:** gradient boosting with Poisson loss on 11 features. On the held-out test period (October–December 2012) it reaches MAE 40.4 and RMSE 64.5 rentals per hour (about 18% of the mean demand).

## Table of Contents

- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
- [API](#api)
- [Docker](#docker)
- [Cloud Deployment](#cloud-deployment)
- [Limitations](#limitations)

## Problem Description

The goal of this project is to predict the total number of bikes rented during a given hour. The target variable is `cnt`, which represents the total number of bike rentals recorded for that hour.

The model uses features such as the hour of the day, day of the week, season, weather conditions, and temperature to estimate rental demand.

This is a **regression task** because the model predicts a numerical value. The goal is to build a model that can estimate bike rental demand from historical data.

## Dataset

This project uses the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), which contains historical bike rental records along with weather and calendar information.

The project uses `hour.csv`, which contains hourly rental data (17,379 rows, 17 columns, two years: 2011 and 2012). It was selected because the goal is to predict bike rental demand for individual hours.

The dataset includes information about rental counts, date and time, season, weather conditions, temperature, humidity, and wind speed. The target variable is `cnt`, representing the total number of bikes rented during each hour.

This dataset is suitable for the project because it combines historical rental demand with time-related and weather-related features, which makes it possible to investigate patterns in bike usage and build a machine learning regression model.

## Exploratory Data Analysis

The exploratory data analysis (notebook `notebooks/eda.ipynb`) focuses on understanding the dataset, checking data quality, and identifying patterns in hourly bike rental demand.

### Data Quality

- The dataset contains 17,379 rows and 17 columns.
- No missing values were found.
- No duplicate records were found.
- The target variable is `cnt`.
- The columns `casual` and `registered` are excluded from model features because their sum equals the target variable.
- Plots revealed additional data issues that missing-value checks do not show; they are listed in the EDA Summary.

### Target Variable Distribution

The target variable `cnt` has a right-skewed distribution. Most hourly rental counts are relatively low, while high rental counts occur less frequently. The mean rental count is approximately 189.46, while the median is 142.

### Hourly Rental Demand

The analysis reveals two distinct peaks in hourly rental demand:

- Morning peak around 8:00.
- Evening peak around 17:00–18:00.

Demand is lowest during the early morning hours. This suggests that bike rentals may be influenced by daily commuting patterns.

### Weekday Rental Demand

The average hourly rental demand remains relatively stable across the week. Thursday and Friday show the highest average rental counts, while Sunday has the lowest. The differences between days are relatively small.

### Seasonal Rental Demand

Average hourly rentals are highest in summer (about 236) and lowest in winter (about 111), with spring (about 208) and fall (about 199) in between. Seasonal differences are more noticeable than differences between weekdays. This shows association, not causation.

### Weather Conditions Analysis

Average bike rental demand decreases as weather conditions become less favorable across the first three weather categories. Clear weather has the highest average rental count, followed by misty or cloudy conditions and light rain or snow.

The Heavy Rain / Snow category contains only 3 observations, so its average rental count is unreliable and should be interpreted with caution.

### Temperature Analysis

The relationship between temperature and hourly bike rentals was explored using a scatter plot, average rentals across temperature ranges, and correlation coefficients. Temperature is normalized in the dataset, so it was converted to approximate degrees Celsius using `temp * 47 - 8`, based on the dataset documentation.

Average hourly rentals increase with temperature, from about 40 rentals below -5 °C to about 331 at 30–35 °C, and level off at around 25–35 °C. The coldest and hottest ranges contain few records (49 and 84), so their averages are less reliable. The correlation between temperature and rentals is positive and moderate (Pearson 0.405, Spearman 0.423), and rentals vary widely at any given temperature. This shows association, not causation.

### Feature Relationships

Relationships between the continuous weather features (`temp`, `atemp`, `hum`, `windspeed`) and the target variable `cnt` were explored using a correlation matrix and scatter plots, and the features were checked for redundancy.

Among the weather features, `temp` and `atemp` have the strongest positive correlation with rentals (about 0.40), `hum` a moderate negative one (-0.32), and `windspeed` a very weak one (0.09). `temp` and `atemp` are almost redundant (correlation 0.99). Plots revealed data issues to handle during data preparation: `atemp` is constant on 2012-08-17 (24 records), `hum` equals 0 on 2011-03-10 (22 records), and `windspeed` equals 0 in about 12.5% of records. These results show association, not causation.

### Yearly Rental Demand

Demand was substantially higher in 2012 (about 235 rentals per hour) than in 2011 (about 144). The cause of the difference is unknown. Because the demand level shifts over time, the strategy for splitting the data into training and test sets needs to be chosen carefully. This shows association, not causation.

### Working Day vs. Non-Working Day Demand

Average hourly rentals are only slightly higher on working days (about 193) than on weekends and holidays (about 181), but the daily pattern is very different: working days have two sharp peaks (around 8:00 and 17:00), while non-working days have a single broad peak around midday. The effect of the hour therefore depends on the day type. This shows association, not causation.

### Holiday Demand

Average hourly rentals are lower on holidays (about 157) than on regular days (about 190), but the holiday group covers only 21 distinct days, so the difference should be interpreted with caution. This shows association, not causation.

### Monthly Rental Demand

Average hourly rentals rise from about 94 in January to about 241 in June, stay at roughly 232–241 from June to September, and then decline to about 142 in December. The lowest month (January) is about 2.5 times lower than the summer plateau. Both years are combined, and demand was higher in 2012. This shows association, not causation.

### EDA Summary

- Hourly demand (`cnt`) is right-skewed (mean about 189, median 142).
- Hour of day shows a pronounced pattern: working days have two peaks (around 8:00 and 17:00–18:00), non-working days have one broad midday peak.
- Demand is highest in summer and lowest in winter, and was substantially higher in 2012 than in 2011 (about 235 vs. 144 rentals per hour).
- Demand increases with temperature and decreases with humidity, but hourly rentals vary widely at any given weather value.
- Data issues to handle during preparation: `atemp` is almost a duplicate of `temp` and is constant on 2012-08-17; `hum` equals 0 on 2011-03-10; `windspeed` equals 0 in about 12.5% of records; some hours are missing; `weathersit = 4` has only 3 records; `holiday = 1` covers only 21 days.
- `casual` and `registered` are excluded as features (target leakage).
- All findings show association, not causation.

## Data Preparation

The data preparation (notebook `notebooks/Data_Preparation.ipynb`) covers data quality decisions, the train/test split, feature selection, feature engineering, and target scaling.

### Data Quality Decisions

- **Zero wind speed (`windspeed = 0`, about 12.5% of records).** Zeros are not concentrated in one year, are more frequent at night, and average rentals within the same hour are close for zero and non-zero wind. They may reflect wind below the measurement threshold or missing measurements, so they are kept unchanged. A `windspeed_zero` flag was kept as a candidate feature and evaluated during model comparison.
- **Zero humidity (`hum = 0`, 22 records on 2011-03-10).** The whole day is affected and zero contradicts the weather description (rain, mist), so these values are treated as missing. They are filled with the median humidity of records with the same `weathersit`. The records are kept. This is an approximation, and the median is calculated on the full dataset (0.13% of records are affected).
- **Constant apparent temperature (`atemp`, 48 records).** A check over all dates found two days, 2012-08-17 and 2012-12-29, where `atemp` is exactly 0.2424 in every hour while `temp` changes, which looks like a recording error. The values are replaced with predictions from a linear model on `temp` fitted on normal records (MAE 0.016). Two other days with a single `atemp` value (2011-09-06, 2012-10-29) are consistent with `temp` and are left unchanged. `atemp` is almost fully explained by `temp` (R² = 0.98).
- **Missing hours (165 of 17,544, about 0.94%).** The hours are not evenly spread: about two thirds of them fall between 0:00 and 5:00 (peak at 3:00–4:00), and 8 days lose 6 or more hours each (for example, 23 hours on 2012-10-29 and 13 on 2012-10-30). Night gaps may partly be hours with zero rentals, but the data cannot confirm this, and the remaining gaps look like outages. Missing hours are not restored and no zero-demand rows are added. Limitation: nighttime demand may be slightly overestimated.
- **Rare categories.** `weathersit = 4` has only 3 records, each a single hour inside an ordinary rainy or misty day, so it is merged into category 3 ("rain or snow, light or heavy"). `holiday = 1` covers 21 days (500 records), all with `workingday = 0`, and 15 of them are Mondays. It is kept unchanged. Limitation: 21 holidays are not enough to model each holiday separately.

### Train/Test Split

The data are split by time, not randomly, because the model is meant to forecast future demand and random splitting would place neighboring hours of the same day in both sets. Training: 2011-01-01 to 2012-09-30 (15,211 records). Test: 2012-10-01 to 2012-12-31 (2,168 records, 12.5%), untouched until the final evaluation. Average rentals are higher in the test set (219.2 vs 185.2) because of the higher demand level in 2012.

Model selection inside the training set uses time-series cross-validation with five folds built on whole days, so no day is shared between the training and validation parts. Metrics are reported per fold. Limitations: the test set covers only October to December, so test metrics may not represent other seasons; the first folds train only on 2011 and cannot show the effect of the year.

### Feature Selection

Columns excluded from the features: `cnt` (target), `casual` and `registered` (target leakage), `instant` and `dteday` (identifiers). 13 candidate features remain: `season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, windspeed_zero`.

- `yr` is kept: 2012 is higher than 2011 in every month, but the gap is not constant (ratio from 1.41 to 2.53 in January–September of the training data). Its usefulness was checked with and without `yr` on the last two cross-validation folds.
- Several candidates overlap: `temp` and `atemp` (correlation 0.992), `season` and `mnth` (season is determined by the month in 89% of records), and `workingday`, which is exactly determined by `weekday` and `holiday`. All of them are kept for tree-based models; for linear models only non-duplicating combinations are used.

### Feature Engineering

Two feature sets are prepared because tree-based and linear models have different needs.

- **Tree-based models** use the 13 candidate features unchanged (later reduced to 11, see Modeling).
- **Linear models** use 10 features: `hr_workingday` (hour and working-day flag combined into 48 categories, because the daily curve has two peaks on working days and one on non-working days), `weathersit`, `mnth_sin` and `mnth_cos` (cyclic month), `temp`, `hum`, `windspeed`, `windspeed_zero`, `holiday`, `yr`. After dummy encoding this gives 57 columns. `weekday` and `atemp` are left out of the base set to avoid exact or near duplication.

### Target Variable and Scaling

The target `cnt` is right-skewed (skewness 1.299 in the training set). Three versions are prepared: `cnt`, `sqrt(cnt)` (skewness 0.296) and `log1p(cnt)` (skewness -0.819, with a left tail from low-demand night hours). The choice is made during model comparison, and all metrics are calculated on the original scale after back-transformation.

Scaling is applied only to linear models: `StandardScaler` on `temp`, `hum`, `windspeed`, `mnth_sin` and `mnth_cos`, fitted on the training part only. Tree-based models use unscaled features.

### Output of This Stage

- `data/processed/train.csv` (15,211 rows) and `data/processed/test.csv` (2,168 rows): dates, 13 features, three engineered columns (`hr_workingday`, `mnth_sin`, `mnth_cos`) and the target `cnt`. The columns `casual` and `registered` are not saved because they leak the target.
- `data/processed/cv_folds.joblib`: five cross-validation folds (row positions in `train.csv`, which must not be re-sorted).
- `models/linear_preprocessor.joblib`: preprocessor of linear models, fitted on the training set.
- The raw file `data/raw/hour.csv` is unchanged.

## Modeling

The modeling notebook is `notebooks/Modeling.ipynb`.

### Evaluation Setup

Models are compared with MAE and RMSE on the original `cnt` scale, computed separately for each of the 5 time-based CV folds. Transformed targets (sqrt, log1p) are converted back before scoring and negative predictions are clipped to 0. The loaded data were checked first: sizes (15,211 + 2,168), no missing values, no leakage columns, and every validation part comes after its training part with no shared days.

### Baselines

Baselines (mean over 5 time-based CV folds, MAE / RMSE): constant 146.3 / 190.7, mean by hour-and-day-type 90.5 / 124.5, linear regression 73.8 / 97.4. Hour and day type explain most of the demand; linear regression is better on average but worse than the hour-and-day-type mean on folds 2 and 3.

### Tree Models

Tree models on all 13 candidate features (mean over 5 CV folds, MAE / RMSE): decision tree 60.5 / 91.5, random forest 54.4 / 81.1, gradient boosting 52.1 / 76.8, versus 73.8 / 97.4 for linear regression. Trees beat the linear model on 4 of 5 folds, but not on fold 3, where training contains only 2011.

### Fold 3 Diagnostics

On the 2011 part of its validation the tree models are the most accurate (gradient boosting MAE 34.6), but on the 2012 part they underestimate demand by about 67–70 rentals per hour (about half of the mean 140.5). The demand level shifted in 2012, which models trained on 2011 only cannot know, so the models are judged mainly on folds 4 and 5, where training includes 2012.

### Target Versions

Target versions (cnt, sqrt, log1p) were compared on folds 4–5. For tree models the raw target is best (gradient boosting MAE 56.3, RMSE 81.8). For linear regression sqrt(cnt) improves MAE from 90.7 to 72.4, while log1p is unstable (it overpredicts on fold 5) and is not used.

### Feature Checks

Feature checks for gradient boosting (mean of folds 4–5): removing `yr` raises MAE from 56.3 to 95.9, so `yr` is essential; `temp` and `atemp` are interchangeable, and `windspeed_zero` duplicates `windspeed == 0` for trees. The tree feature set is reduced to 11 features (no `atemp`, no `windspeed_zero`): MAE 56.6, RMSE 82.1 versus 56.3 and 81.8 with all 13.

The linear model improvements (Ridge, year-by-month interaction, non-linear temperature) were not explored: gradient boosting is clearly ahead (MAE 56.3 vs 72.4 on folds 4–5), so further work focuses on tree models. The linear model with sqrt(cnt) remains as a simple reference.

### Error Analysis

Error analysis of gradient boosting (folds 4–5): once the 2012 level is known (fold 5), the error is about 11–18% of demand in daytime hours. On fold 4 the model underestimates daytime demand by about 25–28% because training contains only two months of 2012, and nearly all of its error is this bias. Nights and rain/snow have the highest relative errors but contribute only about 5% and 8% of the total absolute error.

### Loss Function and Hyperparameters

Poisson loss improved gradient boosting on all five folds in MAE (mean of folds 4–5: MAE 53.7 vs 56.6, RMSE 79.3 vs 82.1), mostly on fold 5; gamma and absolute error were worse than squared error. Poisson loss did not remove the underestimation on fold 4.

Hyperparameter search (25 random combinations, gradient boosting with Poisson loss, 11 features): the best parameters (learning rate 0.064, 575 iterations, 30 leaves, minimum leaf size 24, L2 regularization 5.25) lowered MAE from 70.2 to 65.0 on fold 4 and from 37.2 to 34.5 on fold 5 (RMSE mean of the two 79.3 -> 74.4), and are slightly worse on folds 2 and 3 (by less than 1 MAE). 13 of 25 combinations beat the defaults on both folds. The gain on fold 4 comes mostly from less underestimation, which still makes up most of its error.

### Ensemble Check

The simple average of the tuned gradient boosting and a random forest (11 features) was worse than the tuned gradient boosting alone (mean of folds 4–5: MAE 53.7 vs 49.8, RMSE 79.4 vs 74.4), so no ensemble is used. The final model is gradient boosting with Poisson loss on 11 features.

### Test Results

Test evaluation (2012-10-01 to 2012-12-31, 2,168 rows, mean demand 219.2). Models were trained on the training part only; the test set was used once.

| Model | MAE | RMSE | Bias | MAE / mean demand |
|---|---:|---:|---:|---:|
| Constant | 156.87 | 204.43 | -33.97 | 0.716 |
| Mean by hour and day type | 79.95 | 119.59 | -34.54 | 0.365 |
| Linear regression (sqrt target) | 67.09 | 98.72 | -33.51 | 0.306 |
| **Gradient boosting (final)** | **40.43** | **64.53** | **-14.56** | **0.184** |

(Bias = prediction minus actual; a negative value means underestimation.)

The final model still underestimates demand on average. The test covers three colder months of one year, so the results may not hold for other seasons.

Test error breakdown: relative error is about 15–22% of demand in daytime hours and about 30% at night; the model slightly underestimates demand in all periods (about 5–8% on working days). Rain/snow has the highest relative error (0.46) with only 160 hours (about 7% of test rows, about 11% of the total absolute error). The pattern is similar to the cross-validation analysis.

### Final Model

The final model (`HistGradientBoostingRegressor`, Poisson loss, 11 features: `season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, hum, windspeed`) is refitted on all 17,379 rows and saved to `models/bike_demand_model.joblib` with `models/model_config.json` (features, parameters, training period, scikit-learn version). The reported quality is the test result of the model trained on the training part only; the refitted model has no independent evaluation. `weathersit = 4` must be merged into 3 before prediction.

## Project Structure

```
midterm-project-bike-sharing/
├── data/
│   ├── raw/hour.csv                  # original dataset (unchanged)
│   └── processed/                    # train.csv, test.csv, cv_folds.joblib
├── models/
│   ├── bike_demand_model.joblib      # final model (used by the API)
│   ├── model_config.json             # features, parameters, scikit-learn version
│   └── linear_preprocessor.joblib    # preprocessor of the linear reference model
├── notebooks/
│   ├── eda.ipynb
│   ├── Data_Preparation.ipynb
│   └── Modeling.ipynb
├── train.py                          # retrains and saves the final model
├── predict.py                        # FastAPI service
├── Dockerfile
├── .dockerignore
├── requirements.txt                  # train.py, API and Docker
├── requirements-notebooks.txt        # adds packages used in the notebooks
└── README.md
```

## How to Reproduce

The commands below are for Windows PowerShell and Python 3.11; on Linux or macOS activate the environment with `source .venv/bin/activate`.

1. Create and activate a virtual environment, then install the dependencies. The model was saved with scikit-learn 1.9.1, so this version must not change.

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements-notebooks.txt
   ```

   `requirements.txt` alone is enough to run `train.py` and the API; `requirements-notebooks.txt` includes it and adds the packages used in the notebooks.

2. Run the notebooks in order: `eda.ipynb`, `Data_Preparation.ipynb` (creates `data/processed/` and `models/linear_preprocessor.joblib`), `Modeling.ipynb`. The raw file `data/raw/hour.csv` comes from the UCI dataset.

3. Retrain the final model from the processed files (run from the project root):

   ```
   python train.py
   ```

   The script reads `data/processed/train.csv` and `data/processed/test.csv` (17,379 rows together), validates the data, saves `models/bike_demand_model.joblib` and `models/model_config.json`, and checks that the saved model reloads. With the same parameters it reproduces the notebook result (test MAE 40.425 when trained on the training part only).

4. Start the API locally:

   ```
   uvicorn predict:app
   ```

   Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## API

`predict.py` is a FastAPI service with two endpoints: `GET /health` and `POST /predict` (one record per request). The model is loaded once at startup, and the service refuses to start if the installed scikit-learn version differs from the one recorded in `models/model_config.json`.

**Input.** The 11 model features in coded form (the date is not parsed):

| Field | Values |
|---|---|
| `season` | 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall |
| `yr` | 0 = 2011, 1 = 2012 |
| `mnth` | 1–12 |
| `hr` | 0–23 |
| `holiday` | 0 or 1 |
| `weekday` | 0–6, 0 = Sunday |
| `workingday` | 0 or 1 (1 = not a weekend and not a holiday) |
| `weathersit` | 1 = Clear, 2 = Mist/Cloudy, 3 = Light Rain/Snow, 4 = Heavy Rain/Snow (merged into 3 before prediction) |
| `temp` | normalized temperature, 0–1 (about `temp * 47 - 8` °C) |
| `hum` | normalized humidity, 0–1 |
| `windspeed` | normalized wind speed, 0–1 |

Values outside these ranges are rejected with status 422. `workingday` must be consistent with `weekday` and `holiday` (for example, `weekday = 6` with `workingday = 1` is rejected). The consistency of `season` and `mnth` is not checked.

**Example** (Friday in July 2012, 17:00, clear weather):

```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"season": 3, "yr": 1, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 5, "workingday": 1, "weathersit": 1, "temp": 0.7, "hum": 0.5, "windspeed": 0.2}'
```

The response contains the predicted number of rentals for that hour, rounded to one decimal:

```
{"predicted_cnt": 840.1}
```

**Health check:** `GET /health` returns

```
{"status": "ok", "model_loaded": true, "sklearn_version": "1.9.1"}
```

`yr` accepts only 0 and 1 (2011 and 2012), so the service reproduces the demand level of these two years but cannot predict growth beyond them.

## Docker

Build and run the image from the project root (the image contains `predict.py`, the model files and the pinned dependencies from `requirements.txt`):

```
docker build -t bike-demand-api .
docker run --rm -p 8000:8000 bike-demand-api
```

The service is then available at `http://127.0.0.1:8000/docs`, the same as when run locally. The `data` folder and notebooks are excluded from the image through `.dockerignore`, so `train.py` is run outside the container.

The container was checked with the same three requests as the local service: `/health` returns `status: ok`, the example request returns the same prediction as the local run, and an inconsistent request (`weekday = 6`, `workingday = 1`) returns 422.

## Cloud Deployment

The service is deployed on a virtual machine in Yandex Cloud (Ubuntu 24.04, Docker) and is available at:

**http://84.201.156.91/**

The page at `/` lets you enter the 11 features and get a prediction; the interactive API docs are at `/docs` and the health check at `/health`.

The server holds only the Docker image, not the source code. The image is built on a local machine and transferred to the server as a file:

1. Build the image and save it to a file (on the local machine, in the project root):

   ```bash
   docker build -t bike-demand-api .
   docker save -o bike-demand-api.tar bike-demand-api
   ```

2. Copy `bike-demand-api.tar` to the server over SFTP (for example with FileZilla, using the SSH key of the VM).

3. On the server, load the image and start the container:

   ```bash
   docker load -i bike-demand-api.tar
   docker run -d --name bike-api --restart unless-stopped -p 80:8000 bike-demand-api
   ```

4. Check that the service works:

   ```bash
   docker ps
   curl http://localhost/health
   ```

   The container maps port 80 of the VM to port 8000 of the API; port 80 is opened in the security group of the VM. The container restarts automatically after a server reboot.

To update the deployment, rebuild the image, copy the new file, then run `docker stop bike-api`, `docker rm bike-api`, `docker rmi bike-demand-api`, and repeat steps 3-4.

## Limitations

- The test set covers only three colder months of 2012 (about 92 days) and gives a single estimate without confidence intervals; its metrics may not hold for other seasons. The test was used once and the model was not changed afterwards.
- `yr` takes only the values 0 and 1, so the model cannot forecast later years. The cause of the demand growth in 2012 is unknown.
- Only two cross-validation folds (4 and 5) are reliable for model selection, so the hyperparameter search may be slightly fitted to them.
- The model underestimates demand on average (bias -14.6 on the test set), most noticeably at night on non-working days and in rain or snow; the rain/snow group is small (160 test hours).
- Missing hours were not restored, so night-time demand may be slightly overestimated.
- The final model is refitted on all data and has no independent evaluation.
- The linear reference model was not tuned.
