# LifeStat – Personal Lifestyle Statistical Analyzer

LifeStat is a Python and Streamlit educational project that collects an individual's daily lifestyle observations and applies concepts from Probability and Statistics.

## Features

- Manual daily data entry
- CSV upload, download, and local CSV storage
- Built-in illustrative sample dataset (75 observations)
- Dashboard with lifestyle summaries and trends
- Joint probability of two selected events
- Discrete probability distribution (PMF, expected value, variance, standard deviation)
- Continuous distribution exploration (PDF-style histogram, empirical CDF, interval probability, fitted normal approximation)
- Both regression lines: regression of Y on X and X on Y
- Pearson correlation coefficient
- CSV and simple text report export

## Mathematics covered

1. Joint Probability
2. Discrete Probability Distribution
3. Continuous Probability Distribution
4. Regression Lines
5. Pearson Correlation Coefficient

This project does **not** implement joint discrete/continuous probability distributions, Spearman rank correlation, or machine learning.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Streamlit will show a local URL. Open it in your browser.

## CSV format

The CSV must contain these exact column headers:

```text
date,sleep_hours,exercise_minutes,water_litres,screen_hours,study_hours,energy_rating,mood,sleep_quality
```

Example row:

```text
2026-11-01,7.5,30,2.2,4.0,3.5,8,Good,Good
```

A sample CSV is included in `data/sample_lifestyle_data.csv`.

## Data and privacy

- Data is stored locally in `data/lifestyle_data.csv` when running the app locally.
- The included sample data is synthetic and intended only for demonstration.
- If deployed online, understand the hosting platform's storage and privacy behavior before entering personal information.
- Results describe statistical patterns in the entered data. They are not medical conclusions, and correlation does not establish causation.

## Project structure

```text
LifeStat_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── sample_lifestyle_data.csv
```
