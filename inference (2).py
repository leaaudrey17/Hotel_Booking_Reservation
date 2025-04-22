import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained Random Forest model
model_xg = joblib.load('model_xg1.pkl')

# Streamlit UI elements with updated aesthetic styles
st.markdown("""
    <style>
    body {
        background: linear-gradient(45deg, #f8e2e7, #f5c5d7);  /* Soft pink gradient background */
        font-family: 'Poppins', sans-serif;
        color: #6f3f56;
    }
    .stButton>button {
        background-color: #f6b8c2;  /* Soft pink for buttons */
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 30px;  /* Rounded edges */
        padding: 15px 30px;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: background-color 0.3s ease, transform 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #f29bb6;  /* Darker pink on hover */
        transform: scale(1.05);  /* Slight zoom effect */
    }
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: #fff4f8;  /* Light pink for input fields */
        border-radius: 12px;
        margin-top: 12px;
        padding: 12px;
        font-size: 16px;
        border: 2px solid #f3c6d2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stSelectbox>div>div>input, .stSlider>div>div>input, .stNumberInput>div>div>input {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #f6c6d3;
    }
    .stTitle {
        color: #6f3f56;
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        margin-top: 30px;
        letter-spacing: 1px;
    }
    .stText {
        color: #6f3f56;
        font-size: 18px;
        text-align: center;
        margin-top: 15px;
    }
    .stAlert {
        background-color: #f8e2e7;  /* Soft pastel pink background for alert */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
    }
    .container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 25px;
        margin-top: 30px;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stSlider>div>div>div {
        font-weight: 600;
    }
    .stSelectbox>div>div>div {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header for the app with emoji
st.title('Hotel Booking Cancellation Prediction 🏨❌')

st.markdown('<hr>', unsafe_allow_html=True)

# Description of the app with emoji
st.markdown("""
    <div class="stText">
    Welcome to the Hotel Booking Cancellation Predictor! check if your booking is safe! 🌟
    </div>
""", unsafe_allow_html=True)

# Create a container for the input fields with emojis
with st.container():
    st.subheader('Enter the booking details 📝') 
    
    # Create a grid of input fields inside the container
    col1, col2 = st.columns(2)

    with col1:
        no_of_adults = st.slider('Number of Adults 👨‍👩‍👧‍👦', min_value=0, max_value=5, value=2)
        no_of_children = st.slider('Number of Children 👶', min_value=0, max_value=10, value=0)
        no_of_weekend_nights = st.slider('Number of Weekend Nights 🌅', min_value=1, max_value=8, value=1)
        required_car_parking_space = st.selectbox('Car Parking Space Required 🚗', [0, 1])
        lead_time = st.number_input('Lead Time (days) ⏳', min_value=0, value=10)
        arrival_year = st.selectbox('Arrival Year 📅', [2017, 2018])

    with col2:
        no_of_week_nights = st.slider('Number of Week Nights 🏙️', min_value=0, max_value=7, value=2)
        arrival_month = st.slider('Arrival Month 🌸', min_value=1, max_value=12, value=6)
        arrival_date = st.slider('Arrival Date 📆', min_value=1, max_value=31, value=15)
        repeated_guest = st.selectbox('Repeated Guest 🔄', [0, 1])
        no_of_previous_cancellations = st.slider('Previous Cancellations ❌', min_value=0, max_value=13, value=0)
        no_of_previous_bookings_not_canceled = st.slider('Previous Bookings Not Canceled ✅', min_value=0, max_value=58, value=0)

    st.markdown('<hr>', unsafe_allow_html=True)

    # Create another container for price-related inputs with emojis
    st.subheader('Pricing and Special Requests 💸') 
    
    col3, col4 = st.columns(2)

    with col3:
        avg_price_per_room = st.slider('Average Price per Room 💰', min_value=0.0, max_value=540.0, value=100.0)
        no_of_special_requests = st.slider('Number of Special Requests 🌟', min_value=0, max_value=5, value=0)

    # Encoding categorical features with dropdowns and emojis
    type_of_meal_plan = st.selectbox('Meal Plan Type 🍽️', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    room_type_reserved = st.selectbox('Room Type Reserved 🛏️', ['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
    market_segment_type = st.selectbox('Market Segment Type 📊', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'])

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
    'type_of_meal_plan_Meal Plan 1': [1 if type_of_meal_plan == 'Meal Plan 1' else 0],
    'type_of_meal_plan_Meal Plan 2': [1 if type_of_meal_plan == 'Meal Plan 2' else 0],
    'type_of_meal_plan_Meal Plan 3': [1 if type_of_meal_plan == 'Meal Plan 3' else 0],
    'type_of_meal_plan_Not Selected': [1 if type_of_meal_plan == 'Not Selected' else 0],
    'room_type_reserved_Room_Type 1': [1 if room_type_reserved == 'Room Type 1' else 0],
    'room_type_reserved_Room_Type 2': [1 if room_type_reserved == 'Room Type 2' else 0],
    'room_type_reserved_Room_Type 3': [1 if room_type_reserved == 'Room Type 3' else 0],
    'room_type_reserved_Room_Type 4': [1 if room_type_reserved == 'Room Type 4' else 0],
    'room_type_reserved_Room_Type 5': [1 if room_type_reserved == 'Room Type 5' else 0],
    'room_type_reserved_Room_Type 6': [1 if room_type_reserved == 'Room Type 6' else 0],
    'room_type_reserved_Room_Type 7': [1 if room_type_reserved == 'Room Type 7' else 0],
    'market_segment_type_Aviation': [1 if market_segment_type == 'Aviation' else 0],
    'market_segment_type_Complementary': [1 if market_segment_type == 'Complementary' else 0],
    'market_segment_type_Corporate': [1 if market_segment_type == 'Corporate' else 0],
    'market_segment_type_Offline': [1 if market_segment_type == 'Offline' else 0],
    'market_segment_type_Online': [1 if market_segment_type == 'Online' else 0],
})

# Handle outliers using IQR
numerical_cols = input_data.select_dtypes(include=[np.number]).columns
Q1 = input_data[numerical_cols].quantile(0.25)
Q3 = input_data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
input_data = input_data[~((input_data[numerical_cols] < (Q1 - 1.5 * IQR)) | (input_data[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Standardize numerical features
scaler = StandardScaler()
input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

# Prediction when user clicks "Predict"
if st.button('Predict! 🎉'):
    # Get probabilities for each class (Canceled, Not Canceled)
    prediction_proba = model_xg.predict_proba(input_data)  # Ganti model_rf dengan model_xg

    # Extract the probability for the "Canceled" class (index 1)
    canceled_prob = prediction_proba[0][1]

    # Allow user to adjust the threshold (default 0.5)
    threshold = st.slider('Adjust Threshold for Cancellation Probability 📊', min_value=0.0, max_value=1.0, value=0.5)

    # Display probability and use the threshold to classify
    if canceled_prob >= threshold:
        st.markdown(f"""
            <div class="stAlert">
                <strong>Uh-oh! 😕</strong><br> Your booking might be canceled! 😱 
                <br> Cancellation probability: {canceled_prob * 100:.2f}%<br> 
                Check your details and try again!
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="stAlert">
                <strong>Awesome! 🎉</strong><br> Your booking is safe! 😎 
                <br> Cancellation probability: {canceled_prob * 100:.2f}%<br> 
                Enjoy your stay, no worries!
            </div>
        """, unsafe_allow_html=True)

