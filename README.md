# Forecasting Food Hamper Demand

## Table of Contents
1. [Overview](#overview)
2. [Project Goals](#project-goals)
3. [Key Features](#key-features)
4. [Data Sources](#data-sources)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Development](#model-development)
7. [Model Training](#model-training)
8. [Deployment](#deployment)
9. [Challenges & Solutions](#challenges--solutions)
10. [Future Work](#future-work)
11. [How to Run](#how-to-run)
12. [Conclusion](#conclusion)

## Overview
The **Forecasting Food Hamper Demand** project was developed to support **Islamic Family & Social Services Association (IslamicFamily)** in addressing food insecurity in Edmonton. With food insecurity on the rise, this project utilizes machine learning techniques to forecast food hamper demand, enabling the organization to plan distributions more effectively and allocate resources efficiently.

The primary objective is to ensure that no family goes without essential resources, particularly during peak demand periods, while improving the program’s ability to meet the diverse cultural and demographic needs of the community.

## Project Goals
- **Forecast food hamper demand** using historical data to optimize the allocation of resources.
- **Understand trends** driven by seasonality, economic factors, and cultural events.
- **Enhance program efficiency** and ensure timely support for vulnerable families in Edmonton.
- **Ensure equitable distribution** by predicting demand patterns tied to community needs.

## Key Features
- **Data Collection**: Historical data from November 2022 to August 2024, including client demographics, food hamper pickup details, and scheduling data.
- **Machine Learning Models**: 
  - **ARIMA**: A time-series forecasting model, used initially for trend analysis.
  - **Random Forest**: An ensemble method for capturing non-linear patterns.
  - **XGBoost**: The final choice for modeling due to its superior performance in handling structured data.
  
- **Feature Engineering**: Created lagged features from historical data (e.g., previous pickup counts) to incorporate temporal trends into the model.
- **User Interface**: Deployed a Streamlit-based web application to allow users to interact with the model and get real-time predictions based on input data.

## Data Sources
### Clients Dataset
- Demographic Information: Age, gender, address, dependents.
- Communication Preferences: Contact methods, frequency, and latest interactions.
- Client Status: Active/inactive, created/modified dates, and assigned worker.

### Food Hamper Dataset
- Pickup Details: Scheduled and actual pickup dates, hamper type, and quantities.
- Client Linkages: Information on which clients received which hampers.

These datasets were merged to create a unified dataset, linking food hamper records to their respective clients.

## Data Preprocessing
- **Data Cleaning**: Missing or redundant data was filtered out, and duplicate client records were removed.
- **Time Period Focus**: Data from November 2022 to October 2023 was excluded due to inconsistencies and noise. Analysis focused on data from November 2023 to August 2024 to uncover trends tied to seasonality and cultural events.
- **Feature Selection**: Key columns were retained for analysis, including age, gender, language preferences, scheduled pickup date, and hamper quantities.

## Model Development
### Model Selection
Three models were considered to forecast food hamper demand:
1. **ARIMA**: Simple statistical model for capturing trends and seasonality.
2. **Random Forest**: Ensemble learning model capable of handling complex, non-linear relationships.
3. **XGBoost**: A gradient-boosting algorithm known for its efficiency and high performance on structured datasets.

### Model Training
- **Data Splitting**: The dataset was split into training and test sets while maintaining the temporal order to avoid data leakage.
- **Hyperparameter Tuning**: Grid search was used to fine-tune the hyperparameters of the XGBoost model.
  - Best parameters included:
    - `n_estimators=50`
    - `max_depth=7`
    - `learning_rate=0.1`
    - `subsample=0.8`
    - `min_child_weight=3`
    - `colsample_bytree=1.0`
  
- **Model Evaluation**:
  - **Mean Absolute Error (MAE)**: The tuned model achieved an MAE of 11.72, indicating accurate predictions.
  - **Feature Importance**: Lagged features (pickup counts from previous periods) played a crucial role in forecasting demand.

## Deployment
- **Web Interface**: The model was deployed using Streamlit, a Python library that simplifies the creation of interactive web interfaces. This allows users to input data (e.g., year, month, day, and scheduled count) and receive predictions for future food hamper demand.
- **Model Integration**: The pre-trained XGBoost model was integrated into the Streamlit interface, enabling users to interact with the model via their web browser.

### Steps for Deployment
1. **Train and Save the Model**: The XGBoost model was trained and saved as a pickle file.
2. **Create the Streamlit Interface**: The interface includes dropdowns for selecting dates and a numerical input for the scheduled count.
3. **Launch the App**: The Streamlit app was launched, allowing users to make predictions via an intuitive web interface.

### Monitoring & Maintenance
- **Model Performance Monitoring**: The model's performance will be periodically reviewed, and retraining will be done with new data to ensure continued accuracy.
- **Interface Updates**: The Streamlit interface will be enhanced as needed, based on user feedback and additional feature requests.

## Challenges & Solutions
- **Data Quality**: Some data entry issues were resolved through cleaning and preprocessing techniques.
- **Resource Constraints**: Limited resources for data collection and model training were mitigated by collaborating with stakeholders and using efficient machine learning techniques.

## Future Work
- **Model Retraining**: The model will be periodically retrained with updated data to adapt to new trends and patterns.
- **Feature Expansion**: Additional features, such as weather data or real-time cultural event schedules, could be integrated into future versions to improve prediction accuracy.
- **Enhanced User Interface**: Future iterations of the Streamlit interface may include more advanced features like data visualizations or interactive reports for better decision-making.

## How to Run

## Installation & Usage
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Run the app**:
   ```bash
   python app.py

--

## Conclusion
This project helps Islamic Family & Social Services Association (IslamicFamily) predict and efficiently allocate food hampers to households in need, ultimately improving service delivery and food security within the community. By leveraging machine learning and forecasting, the organization can better anticipate demand and meet the needs of vulnerable families in Edmonton, fostering a more equitable society.


## Acknowledgments
This project was developed for Islamic Family & Social Services Association to enhance their ability to predict and respond to the food hamper needs of Edmonton's food-insecure families.
   
