import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = joblib.load('model_rf.pkl')

def preprocess_input(input_data):
    # Transform input data to dataframe
    df = pd.DataFrame([input_data])

    # Handling missing values (same logic as in your preprocessing)
    df['type_of_meal_plan'] = df['type_of_meal_plan'].fillna(df['type_of_meal_plan'].mode()[0])
    df['avg_price_per_room'] = df['avg_price_per_room'].fillna(df['avg_price_per_room'].median())
    df['required_car_parking_space'] = df['required_car_parking_space'].fillna(0)

    # One hot encoding for categorical variables
    one_hot_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    dummy_frames = []
    for col in one_hot_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        dummy_frames.append(dummies)

    # Concatenate dummy variables to the original dataframe and drop the original categorical columns
    dummies_all = pd.concat(dummy_frames, axis=1)
    df.drop(columns=one_hot_cols, inplace=True)
    df = pd.concat([df, dummies_all], axis=1)

    # Scaling numeric columns
    numerical_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                      'lead_time', 'avg_price_per_room', 'no_of_previous_cancellations',
                      'no_of_previous_bookings_not_canceled', 'no_of_special_requests', 'arrival_year']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Ensure the column order matches what the model expects
    expected_column_order = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 
                             'no_of_week_nights', 'required_car_parking_space', 'lead_time', 
                             'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest', 
                             'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 
                             'avg_price_per_room', 'no_of_special_requests', 
                             'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Meal Plan 3', 
                             'type_of_meal_plan_Not Selected', 'room_type_reserved_Room_Type 2', 
                             'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4', 
                             'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6', 
                             'room_type_reserved_Room_Type 7', 'market_segment_type_Complementary', 
                             'market_segment_type_Corporate', 'market_segment_type_Offline', 
                             'market_segment_type_Online']
    
    # Reorder columns to match the model's expected order
    df = df[expected_column_order]

    return df

# Create Streamlit UI to get input from the user
st.title('Hotel Booking Status Prediction')

# Example inputs (you can customize the form as needed)
input_data = {
    'no_of_adults': st.number_input('Number of Adults', min_value=1, value=1),
    'no_of_children': st.number_input('Number of Children', min_value=0, value=0),
    'no_of_weekend_nights': st.number_input('Number of Weekend Nights', min_value=0, value=1),
    'no_of_week_nights': st.number_input('Number of Week Nights', min_value=0, value=1),
    'lead_time': st.number_input('Lead Time', min_value=0, value=10),
    'avg_price_per_room': st.number_input('Average Price per Room', min_value=0, value=100),
    'no_of_previous_cancellations': st.number_input('Previous Cancellations', min_value=0, value=0),
    'no_of_previous_bookings_not_canceled': st.number_input('Previous Bookings Not Canceled', min_value=0, value=0),
    'no_of_special_requests': st.number_input('Special Requests', min_value=0, value=0),
    'arrival_year': st.number_input('Arrival Year (2017 or 2018)', min_value=2017, max_value=2018, value=2017),
    'type_of_meal_plan': st.selectbox('Meal Plan Type', options=['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3']),
    'room_type_reserved': st.selectbox('Room Type Reserved', options=['Room Type 1', 'Room Type 2']),
    'market_segment_type': st.selectbox('Market Segment Type', options=['Segment 1', 'Segment 2']),
    'required_car_parking_space': st.number_input('Required Car Parking Space (0 or 1)', min_value=0, max_value=1, value=0)  # Added input
}

# Preprocess the input data
processed_data = preprocess_input(input_data)

# Make prediction using the model
if st.button('Predict Booking Status'):
    prediction = model.predict(processed_data)

    # Display the result
    booking_status = 'Not Canceled' if prediction[0] == 0 else 'Canceled'
    st.write(f'Prediction: The booking status is likely to be: {booking_status}')
