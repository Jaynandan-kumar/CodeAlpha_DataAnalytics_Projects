# Exploratory Data Analysis (EDA) Framework

## Overview
This project is a complete **Exploratory Data Analysis (EDA) framework** built using Python.  
It helps in understanding datasets by performing structured data exploration, statistical analysis, visualization, and hypothesis testing.  
The framework follows standard industry practices and automatically generates reports and visual outputs.

---

## Key Features
- Structured **14-step EDA workflow**
- Automatic data cleaning and preprocessing
- Professional data visualizations
- Statistical hypothesis testing
- Outlier and anomaly detection
- Correlation and relationship analysis
- Time series analysis (if date columns exist)
- Customer or data segmentation analysis
- Automated CSV reports and saved plots

---

## Project Structure
EDA_Framework/
├── EDA_framework.py # Main analysis script
├── README.md # Project documentation
├── requirements.txt # Required Python libraries
├── data.csv # (Optional) Input dataset
├── cleaned_dataset.csv # Cleaned data output
├── descriptive_statistics.csv # Statistical summary
├── hypothesis_testing_results.csv # Hypothesis testing results
├── correlation_matrix.csv # Correlation values
└── *.png # Generated plots


---

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt


Or install manually:

pip install pandas numpy matplotlib seaborn missingno scipy

How to Use
Basic Usage

Simply run the script:

python EDA_framework.py

Using Your Own Dataset

Place your CSV file in the project folder

Rename it to data.csv

Run the script

Using a Different File Name

Edit the script and change:

df = pd.read_csv('your_dataset.csv')

What the Framework Does
1. Data Exploration

Loads dataset automatically

Displays structure, shape, and data types

2. Descriptive Statistics

Mean, median, percentiles

Missing values analysis

Skewness and kurtosis

3. Visualization

Histograms and KDE plots

Bar charts and box plots

Category-wise distributions

4. Missing Data Analysis

Missing value counts

Missing data visualization

5. Outlier Detection

IQR-based outlier identification

Box plot visualization

6. Correlation Analysis

Pearson correlation matrix

Heatmap and scatter plots

7. Time Series Analysis

Trend and seasonal analysis (if applicable)

8. Hypothesis Testing

t-test

ANOVA

Correlation tests

Automated interpretation

9. Segmentation Analysis

Group-based data comparison

Segment-wise visualization

10. Reporting

Cleaned dataset export

Statistical reports in CSV

Saved visualization images

Output Files
Generated CSV Files

cleaned_dataset.csv

descriptive_statistics.csv

hypothesis_testing_results.csv

correlation_matrix.csv

Generated Plots

Distribution plots

Missing data analysis

Outlier detection

Correlation heatmaps

Time series analysis

Segmentation analysis

Customization
Change Visualization Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

Adjust Outlier Sensitivity
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

Add Custom Hypotheses
if 'feature1' in df.columns and 'feature2' in df.columns:
    # Perform your statistical test

Common Issues

Module not found

pip install --upgrade pandas numpy matplotlib seaborn missingno scipy


Plots not displaying

import matplotlib
matplotlib.use('TkAgg')


Large dataset performance

df_sample = df.sample(n=1000)

License

This project is licensed under the MIT License.
