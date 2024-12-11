import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from collections import Counter
import folium
import streamlit.components.v1 as components

# Load the dataset with a specified encoding
df_selected = pd.read_csv('df_selected.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('ifssa.png', use_container_width=True)
    st.image('norquest_logo.png', use_container_width=True)

    st.subheader("💡 Abstract:")

    inspiration = '''
    Food insecurity has become a significant challenge in Edmonton, as rising living costs and economic instability have left many families struggling to afford basic necessities. 
    Islamic Family distributes approximately 3,095 food hampers monthly, playing a vital role in alleviating hunger and supporting vulnerable populations. 
    The increasing demand for food hampers, coupled with constrained resources, underscores the importance of predictive tools to anticipate and respond to community needs.
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    This project focuses on using historical data and machine learning to forecast food hamper demand, enabling IslamicFamily to plan for fluctuations and allocate resources efficiently.
    While specific factors like cultural events or seasonal trends will be considered during data exploration, the project remains rooted in evidence-based methodologies to ensure accuracy and cultural relevance.
    With an emphasis on equitable resource distribution, this initiative supports Edmonton's most vulnerable populations, including seniors, immigrant families, and low-income households.
    By combining advanced analytics with a commitment to community well-being, this project seeks to empower IslamicFamily to enhance its operations and uplift the community through compassionate and timely support.
    '''

    st.write(what_it_does)


# Page 2: Exploratory Data Analysis (EDA) 
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")

    # Remove duplicate clients for individual-level analysis
    sub_df = df_selected.drop_duplicates(subset=['unique_client'])

    # --- Categorical Columns ---
    st.header("Categorical Columns Analysis")
    cat_cols = ['Clients_IFSSA.sex', 'Clients_IFSSA.status', 'Clients_IFSSA.household']
    for column in cat_cols:
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=sub_df, x=column, color='#e19a64', 
                           order=sub_df[column].value_counts().index)
        plt.title(f'Distribution of {column}')
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
    num_cols = ['Clients_IFSSA.age', 'Clients_IFSSA.dependents_qty']
    for column in num_cols:
        # Ensure numeric values
        sub_df[column] = pd.to_numeric(sub_df[column], errors='coerce')
        valid_data = sub_df[column].dropna()

        plt.figure(figsize=(8, 6))
        plt.hist(valid_data, bins=10, color='#e19a64', edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        st.pyplot(plt)
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
    map = folium.Map(location=map_center, zoom_start=9)

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



# Page 5: Data Collection
def xai():
    st.title("Explainable AI")
    # st.write("Please fill out the Google form to contribute to our Food Drive!")
    # google_form_url = "https://forms.gle/Sif2hH3zV5fG2Q7P8"#YOUR_GOOGLE_FORM_URL_HERE
    # st.markdown(f"[Fill out the form]({google_form_url})")

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "GeoMapping", "Explainable AI"])

    if app_page == "Dashboard":
        dashboard()
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
