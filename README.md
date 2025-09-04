# Hotel Bookings Dataset Analysis

## Project Overview
This project performs **Exploratory Data Analysis (EDA)**, **data cleaning**, and **feature engineering** on the Hotel Bookings dataset to prepare it for predictive modeling, such as forecasting cancellations. The dataset contains booking information for city and resort hotels, including details like lead time, stay duration, and cancellation status.

The analysis is implemented in a Jupyter notebook (`hotel_bookings_tutorial.ipynb`) using Python and libraries like Pandas, Matplotlib, Seaborn, and Scikit-learn. The project follows three phases: EDA, data cleaning, and preprocessing, ensuring a clean dataset ready for machine learning tasks.

## Dataset
- **Source**: [Hotel Bookings Dataset](https://raw.githubusercontent.com/MaxJoas/hotel_bookings/main/hotel_bookings.csv)
- **Description**: Contains 119,390 bookings with 32 columns, including:
  - `hotel`: Type of hotel (City or Resort).
  - `is_canceled`: Target variable (0 = not canceled, 1 = canceled).
  - `lead_time`: Days between booking and arrival.
  - `adr`: Average daily rate.
  - `country`, `meal`, `market_segment`: Categorical features.
  - Other features like `stays_in_weekend_nights`, `adults`, `children`, etc.
- **Data Issues**: Missing values in `company` (94%), `agent` (14%), `country` (0.4%), `children` (0.003%); outliers in `adr` and `lead_time`; potential data leakage in `reservation_status`.

## Project Structure
- `hotel_bookings_tutorial.ipynb`: Jupyter notebook with the complete analysis pipeline.
- **Phases**:
  1. **EDA & Data Quality Report**:
     - Summary statistics and data structure.
     - Missing values visualization (heatmap).
     - Outlier detection in `adr` and `lead_time` using boxplots and IQR.
  2. **Data Cleaning**:
     - Impute missing values: `company`/`agent` with 0, `country` with mode, `children` with median.
     - Remove duplicates.
     - Cap `adr` outliers at 1000, set negative `adr` to 0.
     - Convert `reservation_status_date` to datetime.
  3. **Feature Engineering & Preprocessing**:
     - New features: `total_guests`, `total_nights`, `is_family`.
     - One-hot encoding for `meal` and `market_segment`.
     - Frequency encoding for `country`, grouping infrequent categories into "Other".
     - Remove leakage columns: `reservation_status`, `reservation_status_date`.
     - Split data into training (80%) and testing (20%) sets.

## Requirements
To run the notebook, install the following Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run

Clone this repository or download the notebook.
Ensure the required libraries are installed.
Open hotel_bookings_tutorial.ipynb in Jupyter Notebook or Google Colab.
Run cells sequentially to perform EDA, cleaning, and preprocessing.
The dataset is loaded directly from a URL, so no local file is needed.

# Key Findings

## Data Quality:
High missing values in company and agent (imputed with 0).
Minor missing values in country and children (imputed with mode/median).
Outliers in adr (e.g., >1000) and lead_time (long tails) addressed by capping.
No significant duplicates found.


## EDA Insights:
Cancellation rate varies by lead_time, market_segment, and hotel.
adr shows extreme values, indicating luxury bookings or errors.


## Preprocessing:
New features enhance interpretability (e.g., is_family).
Encoding balances categorical complexity.
Leakage columns removed to ensure real-world applicability.



# Usage
This project prepares the dataset for tasks like predicting cancellations (is_canceled). The cleaned and engineered dataset (X_train, X_test, y_train, y_test) is ready for model training (e.g., classification models like LightGBM or Logistic Regression).


# Author
Created by Abdulrahman Mahmoud, on September 4, 2025.
