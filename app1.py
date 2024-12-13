import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from collections import Counter
import folium
import streamlit.components.v1 as components
from PIL import Image
import shap

# Load the dataset with a specified encoding
df_selected = pd.read_csv('df_selected.csv', encoding='latin1')

# Page 1: Homepage
def homepage():
    # Add a header banner for visual appeal
    st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                text-align: center;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: semi-bold;
                margin-top: 1rem;
            }
            .footer {
                font-size: 0.8rem;
                text-align: center;
                margin-top: 2rem;
                color: grey;
            }
        </style>
        <h1 class="main-header">📊 Forecasting Food Hamper Demand</h1>
    """, unsafe_allow_html=True)

    # Top Section: Logos side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("ifssa.png", use_container_width=True)
    with col2:
        st.image("norquest_logo.png", use_container_width=True)

    st.write("---")  # Add a horizontal divider

    # Introduction Section
    st.markdown("<h2 class='sub-header'>Introduction</h2>", unsafe_allow_html=True)
    st.info(
        "Food insecurity in Edmonton has surged due to economic instability, leaving families struggling to afford basic necessities. "
        "Islamic Family distributes over **3,095 food hampers monthly**, playing a vital role in supporting vulnerable populations. "
        "This project leverages data analysis and machine learning to forecast demand, ensuring efficient resource allocation."
    )

    # Key Statistics Section
    st.markdown("<h2 class='sub-header'>Food Insecurity at a Glance</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Hampers Distributed", "3,095", "Growing Demand")
    col2.metric("Alberta's Food Insecurity", "27% Higher", "vs National Avg")
    col3.metric("National Food Insecurity", "22.9%", "Canadians Affected")


    st.write("---")

    # Project Overview with Expanders
    st.markdown("<h2 class='sub-header'>Project Goals</h2>", unsafe_allow_html=True)
    with st.expander("✅ Optimize Resource Allocation"):
        st.write("Forecasting demand patterns allows IslamicFamily to better allocate food hampers and plan distributions efficiently.")

    with st.expander("📈 Anticipate Demand Fluctuations"):
        st.write("Using historical data, we analyze trends, seasonality, and cultural events to predict heightened demand periods.")

    with st.expander("🤝 Support Vulnerable Communities"):
        st.write("Focused on immigrant families, seniors, and low-income households, this initiative ensures equitable resource distribution.")

    # Call to Action or Summary at the Bottom
    st.success(
        "By combining advanced analytics with community-focused insights, this project empowers IslamicFamily to create a **timely, equitable, and efficient food hamper program**, ensuring no family is left behind."
    )

    # Footer
    st.markdown("<p class='footer'>Built with ❤️ for the Edmonton community | Forecasting powered by Machine Learning</p>", unsafe_allow_html=True)



# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Page Title
    st.markdown("<h1 style='text-align: center; color: white; font-size: 48px;'>Key Insights</h1>", unsafe_allow_html=True)

    # Horizontal Line
    st.markdown("<hr style='border: 2px solid #F9B233;'>", unsafe_allow_html=True)

    # Layout: Columns
    col1, col2 = st.columns([1, 1])  # Left and right columns for gender
    col3, col4 = st.columns([1, 1])  # Left and right columns for client data
    col5, col6 = st.columns([1, 1])  # Left and right columns for demographic data
    col7, col8 = st.columns([1, 1])  # Left and right columns for dependents and age

    # First Column: Gender Breakdown
    with col1:
        st.metric("Unique Clients", "1,045")
    with col2:
        st.metric("Active Clients", "979")

    # Second Column: Client and Workforce Information
    with col3:
        st.metric("Male", "51.1%")
    with col4:
        st.metric("Female", "48.9%")

    # Grouped Demographic Data (Dependents and Age)
    with col5:
        st.metric("With Dependents", "At Least 1")
    with col6:
        st.metric("Average Age", "~41 Years Old")

    # Third Column: Additional Demographics
    with col7:
        st.metric("Workers", "17")
    with col8:
        st.metric("Belong to a Household", "98%")


    # Remove duplicate clients for individual-level analysis
    sub_df = df_selected.drop_duplicates(subset=['unique_client'])

    # --- Categorical Columns ---
    st.header("Categorical Columns Analysis")
    
    # Plot for 'Clients_IFSSA.sex'
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=sub_df, x='Clients_IFSSA.sex', color='#e19a64', 
                       order=sub_df['Clients_IFSSA.sex'].value_counts().index)
    # Change the x-axis label
    ax.set_xlabel('Sex', fontsize=12)
    plt.title("Distribution of Sex")  # Custom title
    plt.xticks(rotation=45, ha='right')

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    st.pyplot(plt)  # Display plot in Streamlit
    plt.clf()

    # Plot for 'Clients_IFSSA.status'
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=sub_df, x='Clients_IFSSA.status', color='#e19a64', 
                       order=sub_df['Clients_IFSSA.status'].value_counts().index)
    # Change the x-axis label
    ax.set_xlabel('Status', fontsize=12)
    plt.title("Distribution of Client Status")  # Custom title
    plt.xticks(rotation=45, ha='right')

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    st.pyplot(plt)  # Display plot in Streamlit
    plt.clf()

    # Plot for 'Clients_IFSSA.household'
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=sub_df, x='Clients_IFSSA.household', color='#e19a64', 
                       order=sub_df['Clients_IFSSA.household'].value_counts().index)
    # Change the x-axis label
    ax.set_xlabel('Household', fontsize=12)
    plt.title("Distribution of Household")  # Custom title
    plt.xticks(rotation=45, ha='right')

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    st.pyplot(plt)  # Display plot in Streamlit
    plt.clf()

    # --- Numerical Columns ---
    st.header("Numerical Columns Analysis")
    
    # Plot for 'Clients_IFSSA.age'
    sub_df['Clients_IFSSA.age'] = pd.to_numeric(sub_df['Clients_IFSSA.age'], errors='coerce')
    valid_data_age = sub_df['Clients_IFSSA.age'].dropna()

    plt.figure(figsize=(8, 6))
    plt.hist(valid_data_age, bins=10, color='#e19a64', edgecolor='black')
    plt.title("Distribution of Age")  # Custom title
    plt.xlabel('Age')  # Custom axis label
    plt.ylabel('Frequency')

    st.pyplot(plt)  # Display plot in Streamlit
    plt.clf()

    # Plot for 'Clients_IFSSA.dependents_qty'
    sub_df['Clients_IFSSA.dependents_qty'] = pd.to_numeric(sub_df['Clients_IFSSA.dependents_qty'], errors='coerce')
    valid_data_dependents = sub_df['Clients_IFSSA.dependents_qty'].dropna()

    plt.figure(figsize=(8, 6))
    plt.hist(valid_data_dependents, bins=10, color='#e19a64', edgecolor='black')
    plt.title("Distribution of Dependents Quantity")  # Custom title
    plt.xlabel('Dependents Quantity')  # Custom axis label
    plt.ylabel('Frequency')

    st.pyplot(plt)  # Display plot in Streamlit
    plt.clf()


    ##### Counts over time
    # Convert the 'pickup_date' and 'collect_scheduled_date' columns to datetime
    df_selected['pickup_date'] = pd.to_datetime(df_selected['pickup_date'])
    df_selected['collect_scheduled_date'] = pd.to_datetime(df_selected['collect_scheduled_date'])

    # Create an empty date range DataFrame
    date_range = pd.date_range(start='2023-11-01', end='2024-08-28')
    pickup_count = []
    scheduled_count = []

    # Loop through each date in the date range for pickup_date
    for date in date_range:
        pickup_count_value = df_selected[df_selected['pickup_date'].dt.date == date.date()].shape[0]  # Count of rows matching the date
        pickup_count.append(pickup_count_value)

    # Loop through each date in the date range for collect_scheduled_date
    for date in date_range:
        scheduled_count_value = df_selected[df_selected['collect_scheduled_date'].dt.date == date.date()].shape[0]  # Count of rows matching the date
        scheduled_count.append(scheduled_count_value)

    # Create a DataFrame from the results
    df_time_lag = pd.DataFrame({
        'date': date_range,
        'pickup_date_count': pickup_count,
        'collect_scheduled_date_count': scheduled_count
    })

    # Cultural event dates
    cultural_events = {
        'Ramadan Start': '2024-03-10',            # Start of Ramadan
        'Ramadan End': '2024-04-07',              # End of Ramadan
        'Eid al-Adha': '2024-06-17',              # Celebration of Sacrifice
        'Canada Day': '2024-07-01',                # National Day of Canada
        'Islamic New Year': '2024-07-07',   # Islamic New Year
    }

    # Streamlit: Display the plot
    st.title("Pickup and Scheduled Date Counts Over Time")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(
        df_time_lag['date'],
        df_time_lag['pickup_date_count'],
        marker='o',
        label='Pickup Date Count'
        )
    
    plt.plot(
        df_time_lag['date'],
        df_time_lag['collect_scheduled_date_count'],
        marker='o',
        label='Scheduled Date Count'
        )

    # Adding vertical lines for cultural events
    for event_name, event_date in cultural_events.items():
        plt.axvline(pd.to_datetime(event_date), color='red', linestyle='--', label=event_name)
        plt.text(pd.to_datetime(event_date), max(df_time_lag['pickup_date_count'].max(), df_time_lag['collect_scheduled_date_count'].max()) * 0.9, 
                event_name, rotation=90, verticalalignment='center', fontsize=9)


    # Adding titles and labels
    plt.title('Pickup and Scheduled Date Counts Over Time')
    plt.xlabel('Date')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.legend()
    plt.grid()

    # Show plot in Streamlit
    st.pyplot(plt)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")

    # Load time-lagged features and model
    time_lagged_features = pd.read_csv('time_lagged_features.csv')
    model = joblib.load('model_xgb.pkl')
    history_dataset = time_lagged_features.copy()

    # Function to extract seasonality features from a given date
    def extract_seasonality_features(date):
        date_obj = pd.to_datetime(date)
        return {
            'month': date_obj.month,
            'day_of_week': date_obj.weekday(),
            'quarter': date_obj.quarter,
            'day_of_year': date_obj.day_of_year,
            'week_of_year': date_obj.isocalendar()[1],
        }

    # Function to predict the pickup count for a given future date range
    def predict_pickup_for_date_range(target_date, scheduled_count, model_xgb, history_dataset):
        try:
            # Convert target_date to datetime object
            target_date = pd.to_datetime(target_date)

            # Start date is fixed at 2024-08-29
            start_date = pd.to_datetime("2024-08-29")

            # Initialize results container
            results = []

            # Loop through the date range (from start_date to target_date)
            current_date = start_date
            while current_date <= target_date:
                # Feature extraction from the current date
                seasonality = extract_seasonality_features(current_date)

                # Get lag features for the past 7, 14, and 21 days from history_dataset
                lag_7 = history_dataset.loc[history_dataset['date'] == current_date - pd.Timedelta(days=7), 'pickup_date_count'].values
                lag_14 = history_dataset.loc[history_dataset['date'] == current_date - pd.Timedelta(days=14), 'pickup_date_count'].values
                lag_21 = history_dataset.loc[history_dataset['date'] == current_date - pd.Timedelta(days=21), 'pickup_date_count'].values

                # Get scheduled date counts for the past 7, 14, and 21 days
                sched_lag_7 = history_dataset.loc[history_dataset['date'] == current_date - pd.Timedelta(days=7), 'scheduled_date_count'].values
                sched_lag_14 = history_dataset.loc[history_dataset['date'] == current_date - pd.Timedelta(days=14), 'scheduled_date_count'].values
                sched_lag_21 = history_dataset.loc[history_dataset['date'] == current_date - pd.Timedelta(days=21), 'scheduled_date_count'].values

                # Prepare the features for prediction
                features = pd.DataFrame([{
                    'scheduled_date_count': scheduled_count,
                    'month': seasonality['month'],
                    'day_of_week': seasonality['day_of_week'],
                    'quarter': seasonality['quarter'],
                    'day_of_year': seasonality['day_of_year'],
                    'week_of_year': seasonality['week_of_year'],
                    'pickup_date_count_lag_7': lag_7[0] if len(lag_7) > 0 else 0,
                    'scheduled_date_count_7': sched_lag_7[0] if len(sched_lag_7) > 0 else 0,
                    'pickup_date_count_lag_14': lag_14[0] if len(lag_14) > 0 else 0,
                    'scheduled_date_count_14': sched_lag_14[0] if len(sched_lag_14) > 0 else 0,
                    'pickup_date_count_lag_21': lag_21[0] if len(lag_21) > 0 else 0,
                    'scheduled_date_count_21': sched_lag_21[0] if len(sched_lag_21) > 0 else 0,
                }])

                # Ensure features match the model training order
                feature_columns = [
                    'scheduled_date_count', 'pickup_date_count_lag_7', 'scheduled_date_count_7',
                    'pickup_date_count_lag_14', 'scheduled_date_count_14', 'pickup_date_count_lag_21', 'scheduled_date_count_21',
                    'month', 'day_of_week', 'quarter', 'day_of_year', 'week_of_year'
                ]

                features = features[feature_columns]  # Reorder columns to match the model's training data

                # Predict with the trained model
                prediction = model.predict(features)[0]

                # Update historical data with the prediction for the current date
                history_dataset = pd.concat([history_dataset, pd.DataFrame([{
                    'date': current_date,
                    'pickup_date_count': prediction,
                    'scheduled_date_count': scheduled_count,
                }], columns=history_dataset.columns)], ignore_index=True)

                # Store the result for current date
                if current_date == target_date:
                    results.append(f"Predicted pickup count for {current_date.strftime('%Y-%m-%d')} is: {round(prediction)}")

                # Move to the next date
                current_date += pd.Timedelta(days=1)

            return "\n".join(results)  # Join all results to show them at once

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

    # Streamlit App
    st.title("Pickup Count Predictor")

    # Input Interface (on the page, not the sidebar now)
    st.header("Input Parameters")

    # Date picker for the target date
    target_date = st.date_input("Select the Target Date:", pd.to_datetime("2024-09-05"))

    # Numeric input for the scheduled count
    scheduled_count = st.number_input("Enter Scheduled Count:", min_value=0, max_value=1000, value=10)

    # Button to trigger the prediction
    if st.button("Predict"):
        st.write("## Prediction Result")
        prediction_result = predict_pickup_for_date_range(target_date, scheduled_count, model, history_dataset)
        if prediction_result:
            st.success(prediction_result)


# Page 4: Neighbourhood Mapping
def client_mapping():
    st.title("Client GeoMapping in Edmonton")

    # Load the dataset
    clients_df = pd.read_csv('client_cluster.csv', encoding='latin1')

    # Create a folium map centered around the mean latitude and longitude
    map_center = [clients_df['latitude'].mean(), clients_df['longitude'].mean()]
    map = folium.Map(location=map_center, zoom_start=10)

    # Define colors for each cluster
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']

    # Add markers to the map, coloring them by cluster
    for idx, row in clients_df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Client {row['postal_code']}, Cluster {row['cluster']}",
            icon=folium.Icon(color=colors[row['cluster']])
        ).add_to(map)

    # Display the map in Streamlit
    map_html = map._repr_html_()  # Get the HTML representation of the map
    components.html(map_html, height=900)  # Render it in the Streamlit app



# Page 5: Explainable AI
def xai():
    st.title("Explainable AI")

    # Load model and dataset for explainability
    model = joblib.load('model_xgb.pkl')
    time_lagged_features = pd.read_csv('time_lagged_features.csv')

    # Extract the feature columns that were used for the model training
    feature_columns = [
        'scheduled_date_count', 'pickup_date_count_lag_7', 'scheduled_date_count_7',
        'pickup_date_count_lag_14', 'scheduled_date_count_14', 'pickup_date_count_lag_21', 'scheduled_date_count_21',
        'month', 'day_of_week', 'quarter', 'day_of_year', 'week_of_year'
    ]

    # Prepare SHAP explainer
    explainer = shap.Explainer(model)
    
    # Function to get SHAP values for a sample prediction
    def explain_prediction(input_data):
        # Ensure numeric data types and no missing values
        input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        input_data = input_data[feature_columns]  # Ensure it has the correct features
        
        # If the input_data is already a DataFrame, skip the .to_frame() conversion
        if isinstance(input_data, pd.Series):
            input_data = input_data.to_frame().T  # Convert to DataFrame if it's a Series
        
        shap_values = explainer(input_data)
        return shap_values

    # Streamlit input interface for users to select a row to explain
    st.header("Select a Row for Explanation")

    row_index = st.number_input("Select Row Index for Explanation", min_value=0, max_value=len(time_lagged_features)-1, value=0)

    # Get the selected row of data for explanation
    selected_row = time_lagged_features.iloc[[row_index]]

    # Show the selected row's features
    st.write("### Selected Features")
    st.write(selected_row)

    # Get SHAP values for the selected row
    shap_values = explain_prediction(selected_row)

    # Show the SHAP force plot for the selected prediction
    st.write("### SHAP Explanation (Force Plot)")
    shap.initjs()
    st.components.v1.html(shap.force_plot(shap_values[0].base_values, shap_values[0].values, selected_row), height=500)

    # Show the SHAP summary plot (feature importance for the entire dataset)
    st.write("### Feature Importance (Summary Plot)")
    shap.summary_plot(shap_values, time_lagged_features[feature_columns])

    # Optionally, you can show a dependence plot for a specific feature
    feature_name = st.selectbox("Select Feature for Dependence Plot", feature_columns)
    st.write(f"### SHAP Dependence Plot for {feature_name}")
    shap.dependence_plot(feature_name, shap_values, time_lagged_features[feature_columns])

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Homepage", "EDA", "ML Modeling", "GeoMapping", "Explainable AI"])

    if app_page == "Homepage":
        homepage()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "GeoMapping":
        client_mapping()
    elif app_page == "Explainable AI":
        xai()

if __name__ == "__main__":
    main()
