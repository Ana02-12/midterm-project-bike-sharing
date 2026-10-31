# Bike Sharing Demand Prediction

## Project Overview

This project focuses on predicting hourly bike rental demand using machine learning. It uses historical bike rental data along with weather and time-related features to estimate the number of bikes rented during a given hour.

The goal is to build, evaluate, and deploy a machine learning model that can predict hourly rental demand. The project covers the main stages of a machine learning workflow, including exploratory data analysis, data preprocessing, model training, evaluation, and deployment through an API and Docker.

The project uses the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

Accurate demand forecasts can help bike-sharing operators plan bicycle availability, allocate resources more efficiently, and prepare for changes in demand throughout the day.

## Problem Description

The goal of this project is to predict the total number of bikes rented during a given hour. The target variable is `cnt`, which represents the total number of bike rentals recorded for that hour.

The model will use features such as the hour of the day, day of the week, season, weather conditions, and temperature to estimate rental demand.

This is a **regression task** because the model predicts a numerical value. The goal is to build a model that can estimate future bike rental demand using historical data.

## Dataset

This project uses the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), which contains historical bike rental records along with weather and calendar information.

The project uses `hour.csv`, which contains hourly rental data. It was selected because the goal is to predict bike rental demand for individual hours.

The dataset includes information about rental counts, date and time, season, weather conditions, temperature, humidity, and wind speed. The target variable is `cnt`, representing the total number of bikes rented during each hour.

This dataset is suitable for the project because it combines historical rental demand with time-related and weather-related features, allowing us to investigate patterns in bike usage and build a machine learning regression model.

## Exploratory Data Analysis

The initial exploratory data analysis focuses on understanding the dataset, checking data quality, and identifying patterns in hourly bike rental demand.

### Data Quality

- No missing values were found.
- No duplicate records were found.
- The target variable is `cnt`.
- The columns `casual` and `registered` are excluded from model features because their sum equals the target variable.

### Target Variable Distribution

The target variable `cnt` has a right-skewed distribution. Most hourly rental counts are relatively low, while high rental counts occur less frequently.

The mean rental count is approximately 189.46, while the median is 142.

### Hourly Rental Demand

The analysis reveals two distinct peaks in hourly rental demand:

- Morning peak around 8:00.
- Evening peak around 17:00–18:00.

Demand is lowest during the early morning hours. This suggests that bike rentals may be influenced by daily commuting patterns.

### Weekday Rental Demand

This analysis examines how average bike rental demand varies across different days of the week. It helps identify potential differences in rental patterns between weekdays and weekends.
The average hourly rental demand remains relatively stable across the week. Thursday and Friday show the highest average rental counts, while Sunday has the lowest. The differences between days are relatively small.

### Seasonal Rental Demand

This analysis examines how average bike rental demand varies across different seasons of the year. It helps identify seasonal patterns in bike usage and understand how the time of year may influence rental demand.












