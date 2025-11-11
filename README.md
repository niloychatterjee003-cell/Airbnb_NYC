Airbnb NYC Recommender v2 - README
=================================

This project fetches the latest InsideAirbnb NYC datasets, cleans and engineers features, trains a RandomForest model to predict a 'booked' proxy (has >=1 review),
and provides a Streamlit web app that accepts user criteria (price, neighbourhood, room type, dates) and returns recommended available listings with short human explanations.

How to run:
1. pip install -r requirements.txt
2. python backend/fetch_insideairbnb_nyc.py
3. python backend/clean_and_train_v2.py
4. streamlit run frontend/streamlit_app.py

Notes:
- The app uses calendar.csv (if present) to check date availability. If calendar is missing, availability check is skipped.
- The booking label is a proxy based on reviews. For real booking predictions, derive bookings from calendar/reservation data.
