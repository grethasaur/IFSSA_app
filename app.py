import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import folium
import streamlit.components.v1 as components
from PIL import Image
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score

# Load the dataset with a specified encoding
df_selected = pd.read_csv('df_selected.csv', encoding='latin1')

# Page 1: Homepage
def homepage():
    # Add a header banner for visual appeal
    st.markdown("""
        <style>
            .main-header {
                font-size: 6rem;
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
        <h1 class="main-header">🍽️ Forecasting Food Hamper Demand</h1>
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
        "Food insecurity has become a pressing issue across Canada, with Alberta experiencing particularly alarming rates. "
        "Edmonton has been hit hard, with food bank usage doubling since the COVID-19 pandemic. "
        "Additionally, food hamper sizes have decreased due to resource constraints (Tran, 2024). "
        "These challenges underscore the urgent need for innovative solutions to combat the growing food insecurity crisis. "
        "This project uses data analysis and machine learning to forecast demand, helping ensure more efficient resource allocation and better outcomes for those in need."
    )
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Mission:**  
        To forecast and meet community demand across Edmonton by leveraging data-driven insights for the Islamic Family Foundation's food hamper program. 
        We aim to optimize resource allocation, ensuring timely, culturally appropriate support for food-insecure households.
        """)
    
    with col2:
        st.markdown("""
        **Vision:**  
        To create a responsive food aid program that adapts to the community's evolving needs. 
        By utilizing predictive models, we strive to expand the reach and fairness of the Islamic Family Foundation’s resources for all Edmonton residents.
        """)
    
    # Key Statistics Section
    st.markdown("<h2 class='sub-header'>Food Insecurity Statistics at a Glance</h2>", unsafe_allow_html=True)
    col2, col1 = st.columns(2)
    col4, col3 = st.columns(2)

    # National Food Insecurity statistic
    col1.metric("National Food Insecurity", "22.9%", "Canadians Affected")
    col2.metric("Alberta's Food Insecurity", "27%", "vs National Avg")
    
    col3.metric("Newcomer Clients", "32%", "<10 yrs in Canada")

    # IFSSA Specific
    col4.metric("IFSSA Hampers Distributed Monthly", "2000", "Growing Demand")


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
    # Page Title
    st.markdown("""
        <style>
            .main-header {
                font-size: 6rem;
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
        <h1 class="main-header">📊 Visualizations</h1>
    """, unsafe_allow_html=True)
    
    st.header("Key Insights")
    
    # Horizontal Line
    st.markdown("<hr style='border: 2px solid #e19a64;'>", unsafe_allow_html=True)
    
    # Layout: Columns
    col1, col2 = st.columns([1, 1])  # Left and right columns for gender
    col3, col4 = st.columns([1, 1])  # Left and right columns for client data
    col5, col6 = st.columns([1, 1])  # Left and right columns for demographic data
    col7, col8 = st.columns([1, 1])  # Left and right columns for dependents and age

    # First Column: Gender Breakdown
    with col1:
        st.metric("Unique Clients", "1,045")
    with col2:
        st.metric("Active Clients", "986")

    # Second Column: Client and Workforce Information
    with col3:
        st.metric("Top Language Preferred", "Arabic")
    with col4:
        st.metric("Male Clients", "55%")

    # Grouped Demographic Data (Dependents and Age)
    with col5:
        st.metric("With Dependents", "≥ 1")
    with col6:
        st.metric("Average Age", "~41 yrs old")

    # Third Column: Additional Demographics
    with col7:
        st.metric("Workers", "17")
    with col8:
        st.metric("Belong to a Household", "98%")

    # Horizontal Line
    st.markdown("<hr style='border: 2px solid #e19a64;'>", unsafe_allow_html=True)
    
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

    # ---------Plot for 'Clients_IFSSA.household'
    # STEP 1: Split the 'preferred_languages' column by a delimiter
    sub_df['preferred_languages_split'] = sub_df['Clients_IFSSA.preferred_languages'].str.split(',')

    # STEP 2: Flatten the lists and clean up any leading/trailing spaces
    all_languages = [lang.strip() for sublist in sub_df['preferred_languages_split'].dropna() for lang in sublist]

    # STEP 3: Count the occurrences of each language
    language_counts = Counter(all_languages)

    # STEP 4: Convert to a DataFrame to see the counts more easily
    language_counts_df = pd.DataFrame(language_counts.items(), columns=['Language', 'Count'])

    # Sort the DataFrame by 'Count' in descending order and keep only the top 10
    top_languages_df = language_counts_df.sort_values(by='Count', ascending=False).head(10)

    # STEP 5: Create the figure and axes for Streamlit
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot
    bars = ax.bar(top_languages_df['Language'], top_languages_df['Count'], color='#e19a64')
    ax.set_title('Top 10 Language Preferences Count', fontsize=14)
    ax.set_xlabel('Language', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add count labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
                ha='center', va='bottom', fontsize=10)

    # Adjust layout and render in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    

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

    #Tableu Dashboard
    #Title and clickable link to Tableau Interactive Dashboard
    st.title("Tableau Interactive Dashboard")
    st.markdown("[Click here to view the Tableau Dashboard]"
                "(https://public.tableau.com/views/WorkIntegratedllDashboard/Dash1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)"
               )

# Page 3: Neighbourhood Mapping
def client_mapping():
    st.markdown("""
        <style>
            .main-header {
                font-size: 6rem;
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
        <h1 class="main-header">🗺️ Client GeoMapping in Edmonton</h1>
    """, unsafe_allow_html=True)
    
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


# Page 4: Machine Learning Modeling
def machine_learning_modeling():
    st.markdown("""
        <style>
            .main-header {
                font-size: 6rem;
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
        <h1 class="main-header">📈 Machine Learning Modelling</h1>
    """, unsafe_allow_html=True)
    
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
    st.header("Pickup Count Predictor")

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


# Page 5: Explainable AI
def xai():
    st.markdown("""
        <style>
            .main-header {
                font-size: 6rem;
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
    <h1 class="main-header">🧠 Explainable AI</h1>
    """, unsafe_allow_html=True)

    # Introduction to Explainable AI
    st.write("""
    **Explainable AI (XAI)** techniques help to interpret how machine learning models make predictions. 
    In this section, we will explore methods to understand the model's decision-making process and how it uses different features for prediction.
    This improves trust in the model and makes the results more actionable for users.
    """)

    # SHAP Values
    st.subheader("SHAP Values")
    st.write("""
    SHAP (SHapley Additive exPlanations) values are used to explain the contribution of each feature 
    to individual predictions. By calculating SHAP values, we can see how much each feature 
    (such as `scheduled_date_count`, `month`, `day_of_week`, etc.) contributes to the prediction for a specific data point.
    """)
        
   # Below is the SHAP analysis for a sample prediction:
    
    # Load time-lagged features and model
    time_lagged_features = pd.read_csv('time_lagged_features.csv')
    model = joblib.load('model_xgb.pkl')
    history_dataset = time_lagged_features.copy()
    feature_columns = [
    'scheduled_date_count', 'pickup_date_count_lag_7', 'scheduled_date_count_7',
    'pickup_date_count_lag_14', 'scheduled_date_count_14', 'pickup_date_count_lag_21', 'scheduled_date_count_21',
    'month', 'day_of_week', 'quarter', 'day_of_year', 'week_of_year'
    ]

    # Initialize SHAP explainer for XGBoost model
    explainer = shap.Explainer(model)
    
    # A sample data point from history_dataset for explanation
    sample_data = history_dataset.iloc[0][feature_columns].values.reshape(1, -1)
    
    # Calculate SHAP values
    shap_values = explainer(sample_data)
    
    # Display SHAP summary plot
    shap.summary_plot(shap_values, sample_data, feature_names=feature_columns)
    fig = plt.gcf()  # Get the current figure from SHAP plot
    st.pyplot(fig)  # Pass the figure explicitly to st.pyplot()

    # Feature Importance Plot (using SHAP or model feature importances)
    st.subheader("Feature Importance")
    st.write("""
    Feature importance plots show the relative importance of each feature in making predictions. 
    The more important a feature is, the larger its contribution to the final model decision.
    Below is the feature importance plot for the current model:
    """)

    # Get feature importances from the XGBoost model
    feature_importance = model.feature_importances_
    
    # Create a DataFrame for feature importances and sort them
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Display feature importance as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis
    ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    # Residual Analysis Against Time
    st.header("Residual Analysis Against Time")
    st.write("""
    Plotting residuals against time helps identify temporal patterns, trends, or periods where the model may underperform.
    Ideally, residuals should be randomly distributed across time.
    """)
    
    # Generate a date range with the same number of rows as the history_dataset
    history_dataset['date'] = pd.date_range(start='2023-11-01', periods=len(history_dataset), freq='D')
    
    # Set 'DATE' as the index
    history_dataset = history_dataset.set_index('date').sort_index()

    # Predictions and residuals
    predictions = model.predict(history_dataset[feature_columns])
    residuals = history_dataset['pickup_date_count'] - predictions

    # Add residuals and predictions back to dataset
    history_dataset['Predicted'] = predictions
    history_dataset['Residuals'] = residuals

    # Residuals vs. Time Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use 'day_of_year' or 'week_of_year' as a proxy for time
    ax.plot(history_dataset.index, residuals, label='Residuals', color='tab:blue')
    ax.set_xlabel('Day of Year')  # Update x-axis label
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Over Time')
    st.pyplot(fig)

    # Actual vs. Predicted Plot
    st.header("Actual vs Predicted Pickup Counts")
    st.write("""
    Comparing actual vs. predicted values helps visualize the model's accuracy and alignment with observed data.
    Below is the plot of actual vs. predicted pickup counts:
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_dataset.index, history_dataset['pickup_date_count'], label='Actual', color='tab:green')
    ax.plot(history_dataset.index, history_dataset['Predicted'], label='Predicted', color='tab:orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Pickup Counts')
    ax.set_title('Actual vs Predicted Pickup Counts Over Time')
    ax.legend()
    st.pyplot(fig)

    # R² Score Calculation and Display
    st.subheader("R² Score")
    r2 = r2_score(history_dataset['pickup_date_count'], predictions)
    st.write(f"The R² score for the model is: **{r2:.3f}**")

    # Conclusion
    st.write("""
    In this section, we explore various XAI techniques to help us understand how the model makes predictions:
    - **SHAP Values**: Show how much each feature contributes to specific predictions.
    - **Feature Importance**: Displays the relative importance of each feature in model decisions.
    - **Residual Analysis**: Analyzes the residuals (actual vs. predicted values) for model performance evaluation.
    
    By incorporating these techniques, we ensure transparency and interpretability in the model, building trust in the results.
    """)



# Main App Logic
def main():
    st.sidebar.title("IFSSA Food Hamper App")
    app_page = st.sidebar.radio("Select a Page", ["Homepage", "EDA","GeoMapping", "ML Modeling", "Explainable AI"])

    if app_page == "Homepage":
        homepage()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "GeoMapping":
        client_mapping()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Explainable AI":
        xai()

if __name__ == "__main__":
    main()
