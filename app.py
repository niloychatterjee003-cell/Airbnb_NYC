# app.py
"""
Airbnb NYC Recommender (Streamlit)
- Uses public NYC dataset (AB_NYC_2019) from GitHub raw URL
- Provides EDA, trains a quick model (booked proxy), and recommends listings by user filters
- Ready to deploy on Streamlit Cloud (save as app.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

st.set_page_config(page_title="Airbnb NYC Recommender", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/dgomonov/new-york-city-airbnb-open-data/master/AB_NYC_2019.csv"

@st.cache_data(ttl=60*60*6)
def load_data():
    """Download and load the public CSV."""
    res = requests.get(RAW_URL, timeout=30)
    res.raise_for_status()
    df = pd.read_csv(StringIO(res.text))
    return df

@st.cache_data(ttl=60*60*6)
def preprocess(df):
    """Clean and create useful fields."""
    d = df.copy()
    # normalize price
    d['price'] = pd.to_numeric(d['price'], errors='coerce')
    # basic features
    d['neighbourhood_final'] = d.get('neighbourhood', d.get('neighbourhood_group', 'Unknown')).fillna('Unknown')
    d['room_type_final'] = d['room_type'].fillna('Unknown')
    d['num_amenities'] = 0  # dataset doesn't include amenities; keep placeholder
    d['has_wifi'] = 0
    # reviews -> booking proxy
    d['num_reviews'] = pd.to_numeric(d.get('number_of_reviews', 0), errors='coerce').fillna(0).astype(int)
    d['reviews_per_month'] = pd.to_numeric(d.get('reviews_per_month', 0), errors='coerce').fillna(0.0)
    d['booked_proxy'] = (d['num_reviews'] > 0).astype(int)
    # ensure id column
    if 'id' not in d.columns:
        d = d.reset_index().rename(columns={'index':'id'})
    return d

@st.cache_resource
def train_model(df):
    """Train a simple RF classifier to predict booked_proxy."""
    dfm = df.copy()
    # keep rows with price
    dfm = dfm[dfm['price'].notnull()].copy()
    features = ['price','room_type_final','neighbourhood_final','reviews_per_month']
    features = [f for f in features if f in dfm.columns]
    X = dfm[features]
    y = dfm['booked_proxy']
    # simple fill
    X['price'] = X['price'].fillna(X['price'].median())
    X['reviews_per_month'] = X['reviews_per_month'].fillna(0.0)
    # split guard
    if y.nunique() < 2 or len(y) < 200:
        return None, features
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
    ])
    pipe = Pipeline([('pre', pre), ('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    return pipe, features

def human_reasons(row, min_price, max_price):
    """Generate short human-friendly reasons for recommendation."""
    reasons = []
    price = row.get('price', np.nan)
    if pd.notnull(price):
        if price <= (min_price + (max_price-min_price)*0.33):
            reasons.append("Competitive price for your range")
        elif price >= (min_price + (max_price-min_price)*0.75):
            reasons.append("Premium-priced option")
    if row.get('num_reviews',0) > 50:
        reasons.append("Many reviews — popular choice")
    if row.get('reviews_per_month',0) > 1.0:
        reasons.append("Consistently reviewed recently")
    # availability proxy (not exact without calendar)
    if 'availability_365' in row and pd.notnull(row['availability_365']):
        if row['availability_365'] < 60:
            reasons.append("High demand (low availability)")
        else:
            reasons.append("Good availability")
    if not reasons:
        reasons.append("Matches your filters")
    return "; ".join(reasons)

# ---------------------------
# App layout
# ---------------------------

st.title("🏠 Airbnb NYC — Recommender & EDA")
st.markdown("This app fetches a public Airbnb NYC dataset and provides EDA + a simple recommender. "
            "The booking label is a proxy (listing has ≥1 review).")

# Load & preprocess
with st.spinner("Downloading dataset..."):
    raw = load_data()
    df = preprocess(raw)

# Sidebar filters
st.sidebar.header("Search & Filters")
min_price, max_price = st.sidebar.slider("Price range (USD)", 0, 1000, (50, 400), step=5)
neighbourhood_sel = st.sidebar.selectbox("Neighbourhood (or All)", ["All"] + sorted(df['neighbourhood_final'].dropna().unique().tolist()))
room_type_sel = st.sidebar.selectbox("Room type", ["Any"] + sorted(df['room_type_final'].dropna().unique().tolist()))
limit_results = st.sidebar.number_input("Max results to show", min_value=5, max_value=50, value=10)

# EDA section
if st.checkbox("Show EDA (price distribution, top neighborhoods)"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price distribution")
        hist_df = df['price'].dropna()
        st.bar_chart(pd.cut(hist_df, bins=40).value_counts().sort_index())
    with col2:
        st.subheader("Top neighbourhoods by listing count")
        topn = df['neighbourhood_final'].value_counts().head(15)
        st.bar_chart(topn)

    st.subheader("Summary stats")
    st.write(df[['price','num_reviews','reviews_per_month','availability_365']].describe().T)

# Train or load model
with st.spinner("Training quick internal model..."):
    model, model_features = train_model(df)

# Filter candidates based on user inputs
candidates = df[df['price'].between(min_price, max_price)].copy()
if neighbourhood_sel != "All":
    candidates = candidates[candidates['neighbourhood_final'].str.contains(neighbourhood_sel, case=False, na=False)]
if room_type_sel != "Any":
    candidates = candidates[candidates['room_type_final'] == room_type_sel]

st.markdown(f"### Found **{len(candidates)}** listings matching price & filters")

# Score candidates
def score_row_apply(r):
    # If trained model exists, use that predicted probability
    if model is not None:
        # prepare row for model input
        try:
            Xr = pd.DataFrame([{f: r.get(f, None) for f in model_features}])
            # ensure numeric col types exist
            for c in Xr.columns:
                if Xr[c].dtype == 'object' and pd.isna(Xr[c].iloc[0]):
                    Xr[c] = Xr[c].fillna('unknown')
            prob = model.predict_proba(Xr)[0][1]
        except Exception:
            prob = 0.0
    else:
        # heuristic scoring without model
        price = r.get('price', 1e6)
        pscore = 1 - (price - min_price) / max(1, (max_price - min_price))
        avail_pct = r.get('availability_365', 0.5) / 365.0
        amen = min(r.get('num_amenities',0), 10) / 10.0
        prob = max(0, min(1, 0.4*pscore + 0.4*(avail_pct) + 0.2*amen))
    return prob

if not candidates.empty:
    candidates['pred_score'] = candidates.apply(score_row_apply, axis=1)
    # add reason text
    candidates['reason_text'] = candidates.apply(lambda r: human_reasons(r, min_price, max_price), axis=1)
    candidates = candidates.sort_values('pred_score', ascending=False)
    topk = candidates.head(limit_results).copy()
    display_cols = ['id','name','host_id','neighbourhood_final','room_type_final','price','num_reviews','reviews_per_month','pred_score','reason_text']
    display_cols = [c for c in display_cols if c in topk.columns]
    st.subheader("Top recommendations")
    st.dataframe(topk[display_cols].reset_index(drop=True).style.format({'pred_score':'{:.3f}', 'price':'${:,.0f}'}))

    st.markdown("#### Detail cards")
    for _, row in topk.head(5).iterrows():
        title = row.get('name', f"Listing {int(row['id'])}")
        price_s = f"${row.get('price', 'N/A')}"
        neigh = row.get('neighbourhood_final', 'Unknown')
        room = row.get('room_type_final', '')
        score = row.get('pred_score', 0.0)
        reason = row.get('reason_text', '')
        st.markdown(f"**{title}** — {room} — {neigh} — **{price_s}**  \n**Why:** {reason}  \n**Predicted score:** {score:.3f}")
        st.markdown("---")
else:
    st.warning("No listings found. Try widening price range or change neighbourhood/room type.")

st.markdown("---")
st.markdown("**Notes:** This app uses a *booking-proxy* (listing has ≥1 review) to approximate bookings. For true booking predictions, calendar/reservation data is required.")
