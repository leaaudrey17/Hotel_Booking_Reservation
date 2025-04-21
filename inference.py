import streamlit as st
import pandas as pd
import numpy as np

# Input data dari user
no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, value=2)
no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Jumlah Malam Akhir Pekan", min_value=0, value=1)
no_of_week_nights = st.number_input("Jumlah Malam Hari Kerja", min_value=0, value=2)
type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
required_car_parking_space = st.selectbox("Butuh Parkir Mobil?", [0, 1])
room_type_reserved = st.selectbox("Tipe Kamar", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
lead_time = st.number_input("Lead Time (hari sebelum menginap)", min_value=0, value=10)
arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
arrival_month = st.number_input("Bulan Kedatangan (1-12)", min_value=1, max_value=12, value=1)
arrival_date = st.number_input("Tanggal Kedatangan (1-31)", min_value=1, max_value=31, value=1)
market_segment_type = st.selectbox("Tipe Market", ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
repeated_guest = st.selectbox("Guest Kembali?", [0, 1])
avg_price_per_room = st.number_input("Rata-rata Harga per Kamar", min_value=0.0, value=100.0)
no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Jumlah Pemesanan Sebelumnya yg Tidak Dibatalkan", min_value=0, value=0)
no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, value=0)

# One-hot encoding manual (harus sama dengan preprocessing training)

# Label encoding untuk variabel dengan 2 kelas
required_car_parking_space = int(required_car_parking_space)
repeated_guest = int(repeated_guest)

# One-hot encoding manual untuk variabel kategori
meal_plan_columns = ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3']
meal_plan = [1 if type_of_meal_plan == mp else 0 for mp in meal_plan_columns]

room_type_columns = ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
room_type = [1 if room_type_reserved == rt else 0 for rt in room_type_columns]

market_segment_columns = ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary']
market_segment = [1 if market_segment_type == ms else 0 for ms in market_segment_columns]

# Membuat dataframe untuk input yang sudah di-encode
input_data = {
    'no_of_adults': no_of_adults,
    'no_of_children': no_of_children,
    'no_of_weekend_nights': no_of_weekend_nights,
    'no_of_week_nights': no_of_week_nights,
    'required_car_parking_space': required_car_parking_space,
    'lead_time': lead_time,
    'arrival_year': 0 if arrival_year == 2017 else 1,
    'avg_price_per_room': avg_price_per_room,
    'no_of_previous_cancellations': no_of_previous_cancellations,
    'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
    'no_of_special_requests': no_of_special_requests,
}

# Gabungkan kolom one-hot encoding ke dalam input_data
input_data.update(dict(zip(meal_plan_columns, meal_plan)))
input_data.update(dict(zip(room_type_columns, room_type)))
input_data.update(dict(zip(market_segment_columns, market_segment)))

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediksi
if st.button("Prediksi"):
    # Untuk demo, misalnya model prediksi acak (untuk testing)
    prediction = np.random.choice([0, 1])
    status = "Canceled" if prediction == 1 else "Not Canceled"
    st.success(f"Status Booking: **{status}**")
