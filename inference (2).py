import streamlit as st
import joblib
import pandas as pd

# Load the trained Random Forest model
model_rf = joblib.load('model_xg.pkl')

# Streamlit UI elements
st.title('Hotel Booking Cancellation Prediction')
st.write('This app predicts whether a hotel booking will be canceled or not based on the input features.')

# Input fields for the user to provide data (matching the features from the model)
no_of_adults = st.number_input('Number of Adults', min_value=1, max_value=10, value=2)
no_of_children = st.number_input('Number of Children', min_value=0, max_value=5, value=0)
no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, max_value=7, value=1)
no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, max_value=7, value=2)
required_car_parking_space = st.selectbox('Car Parking Space Required', [0, 1])
lead_time = st.number_input('Lead Time (days)', min_value=0, value=10)
arrival_year = st.number_input('Arrival Year', min_value=2000, max_value=2025, value=2023)
arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=6)
arrival_date = st.number_input('Arrival Date', min_value=1, max_value=31, value=15)
repeated_guest = st.selectbox('Repeated Guest', [0, 1])
no_of_previous_cancellations = st.number_input('Number of Previous Cancellations', min_value=0, max_value=13, value=0)
no_of_previous_bookings_not_canceled = st.number_input('Number of Previous Bookings Not Canceled', min_value=0, max_value=58, value=0)
avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, max_value=540.0, value=100.0)
no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, max_value=5, value=0)

# Encoding categorical features (e.g., meal plan, room type, market segment)
type_of_meal_plan = st.selectbox('Meal Plan Type', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
room_type_reserved = st.selectbox('Room Type Reserved', ['Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
market_segment_type = st.selectbox('Market Segment Type', ['Complementary', 'Corporate', 'Offline', 'Online'])

# Prepare the input data for the model
input_data = pd.DataFrame({
    'no_of_adults': [no_of_adults],
    'no_of_children': [no_of_children],
    'no_of_weekend_nights': [no_of_weekend_nights],
    'no_of_week_nights': [no_of_week_nights],
    'required_car_parking_space': [required_car_parking_space],
    'lead_time': [lead_time],
    'arrival_year': [arrival_year],
    'arrival_month': [arrival_month],
    'arrival_date': [arrival_date],
    'repeated_guest': [repeated_guest],
    'no_of_previous_cancellations': [no_of_previous_cancellations],
    'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
    'avg_price_per_room': [avg_price_per_room],
    'no_of_special_requests': [no_of_special_requests],
    'type_of_meal_plan_Meal Plan 2': [1 if type_of_meal_plan == 'Meal Plan 2' else 0],
    'type_of_meal_plan_Meal Plan 3': [1 if type_of_meal_plan == 'Meal Plan 3' else 0],
    'type_of_meal_plan_Not Selected': [1 if type_of_meal_plan == 'Not Selected' else 0],
    'room_type_reserved_Room_Type 2': [1 if room_type_reserved == 'Room Type 2' else 0],
    'room_type_reserved_Room_Type 3': [1 if room_type_reserved == 'Room Type 3' else 0],
    'room_type_reserved_Room_Type 4': [1 if room_type_reserved == 'Room Type 4' else 0],
    'room_type_reserved_Room_Type 5': [1 if room_type_reserved == 'Room Type 5' else 0],
    'room_type_reserved_Room_Type 6': [1 if room_type_reserved == 'Room Type 6' else 0],
    'room_type_reserved_Room_Type 7': [1 if room_type_reserved == 'Room Type 7' else 0],
    'market_segment_type_Complementary': [1 if market_segment_type == 'Complementary' else 0],
    'market_segment_type_Corporate': [1 if market_segment_type == 'Corporate' else 0],
    'market_segment_type_Offline': [1 if market_segment_type == 'Offline' else 0],
    'market_segment_type_Online': [1 if market_segment_type == 'Online' else 0],
})

# Prediction when the user clicks the button
if st.button('Predict'):
    prediction = model_rf.predict(input_data)
    st.write(f'Predicted Booking Status: {"Not Canceled" if prediction[0] == "Not_Canceled" else "Canceled"}')
