import streamlit as st
import pickle
import pandas as pd
import numpy as np

# List of IPL teams
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

# List of cities where IPL matches are played
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set up the Streamlit app
st.title('IPL Win Predictor')

# Create two columns for team selection
col1, col2 = st.columns(2)

# Select batting team
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

# Select bowling team
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Select the host city
selected_city = st.selectbox('Select host city', sorted(cities))

# Input the target score
target = st.number_input('Target', min_value=0, step=1)

# Create three columns for match statistics
col3, col4, col5 = st.columns(3)

# Input current score
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)

# Input completed overs
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)

# Input wickets fallen
with col5:
    wickets = st.number_input('Wickets fallen', min_value=0, max_value=10, step=1)

# Button to trigger prediction
if st.button('Predict Probability'):
    # Calculate required statistics
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 20

    # Calculate percentage features
    wickets_left_percentage = wickets_left / 10
    balls_left_percentage = balls_left / 120
    runs_left_percentage = runs_left / target if target > 0 else 0

    # Create input dataframe for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr],
        'wickets_left_percentage': [wickets_left_percentage],
        'balls_left_percentage': [balls_left_percentage],
        'runs_left_percentage': [runs_left_percentage]
    })

    # Make prediction
    result = pipe.predict_proba(input_df)
    loss_prob = result[0][0]
    win_prob = result[0][1]

    # Display results
    st.header(f"{batting_team} - {round(win_prob * 100)}%")
    st.header(f"{bowling_team} - {round(loss_prob * 100)}%")