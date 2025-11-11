\
import streamlit as st
import pandas as pd, os, subprocess, sys, joblib
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, 'data')
OUT_DIR = os.path.join(BASE, 'outputs')
BACK = os.path.join(BASE, 'backend')

st.set_page_config(page_title='Airbnb NYC Recommender (v2)', layout='wide')
st.title('Airbnb NYC — Smart Recommender (v2)')
st.markdown('Use the form to search for available rooms and get AI-assisted recommendations and explanations.')

col1, col2 = st.columns([2,1])

with col2:
    if st.button('Fetch latest InsideAirbnb NYC data'):
        st.info('Fetching... this can take up to a minute.')
        subprocess.run([sys.executable, os.path.join(BACK, 'fetch_insideairbnb_nyc.py')])
        st.success('Fetch attempted. Check data/ folder.')
    if st.button('Clean & Train (auto)'):
        st.info('Cleaning data and training model...')
        subprocess.run([sys.executable, os.path.join(BACK, 'clean_and_train_v2.py')])
        st.success('Done. Check outputs/ for cleaned data and model.')

with col1:
    with st.form('search'):
        st.subheader('Search criteria')
        min_price, max_price = st.slider('Price range (USD)', 0, 2000, (50, 400))
        neighbourhood = st.text_input('Neighbourhood (leave blank for any)', value='')
        room_type = st.selectbox('Room type', options=['Any','Entire home/apt','Private room','Shared room','unknown'])
        checkin = st.date_input('Check-in date', value=datetime.today().date() + timedelta(days=7))
        checkout = st.date_input('Check-out date', value=checkin + timedelta(days=3))
        submit = st.form_submit_button('Find rooms & Recommend')
    if submit:
        # load cleaned listings if available, otherwise try raw listings
        cleaned = os.path.join(OUT_DIR, 'cleaned_listings_v2.csv')
        if os.path.exists(cleaned):
            df = pd.read_csv(cleaned)
        else:
            raw = os.path.join(DATA_DIR, 'listings.csv')
            if os.path.exists(raw):
                df = pd.read_csv(raw, low_memory=False)
                st.warning('Using raw listings; for best results run Clean & Train first.')
            else:
                st.error('No listings found. Fetch data first.'); st.stop()
        # normalize fields
        df['price'] = pd.to_numeric(df['price'], errors='coerce') if 'price' in df.columns else pd.to_numeric(df.get('price', pd.Series([None]*len(df))), errors='coerce')
        # filter by price and room type and neighbourhood
        filt = df['price'].between(min_price, max_price)
        if room_type != 'Any' and 'room_type_final' in df.columns:
            filt = filt & (df['room_type_final'] == room_type)
        if neighbourhood.strip() != '':
            if 'neighbourhood_final' in df.columns:
                filt = filt & (df['neighbourhood_final'].str.contains(neighbourhood, case=False, na=False))
        candidates = df[filt].copy()
        st.write(f'Found {len(candidates)} listings matching price/filters. Now checking availability for your dates...')
        # check availability using calendar if present in outputs/data folder
        cal_path = os.path.join(DATA_DIR, 'calendar.csv')
        available_ids = set()
        if os.path.exists(cal_path):
            cal = pd.read_csv(cal_path, parse_dates=['date'], low_memory=False)
            # determine availability per listing for date range
            mask_dates = (cal['date'] >= pd.to_datetime(checkin)) & (cal['date'] < pd.to_datetime(checkout))
            cal_window = cal[mask_dates]
            # standardize available column detection
            avail_col = next((c for c in cal_window.columns if 'avail' in c.lower()), None)
            id_col = next((c for c in cal_window.columns if 'listing' in c.lower()), None)
            if avail_col and id_col:
                grouped = cal_window.groupby(id_col)[avail_col].agg(lambda s: all([str(x).lower() in ['t','true','y','yes','1','available'] for x in s]))
                available_ids = set(grouped[grouped==True].index.astype(int).tolist())
            else:
                st.warning('Calendar file present but columns not recognized; skipping date availability check.')
        else:
            st.info('No calendar.csv found; availability check skipped (will show all matched listings).')
        # keep only available ids if calendar used
        if len(available_ids)>0:
            candidates = candidates[candidates['id'].isin(list(available_ids))]
            st.write(f'{len(candidates)} listings are available for your date range.')
        # if no candidates left
        if candidates.empty:
            st.warning('No available listings matched your criteria. Try widening the price range or changing dates.')
        else:
            # load model
            model_path = os.path.join(OUT_DIR, 'rf_booked_proxy_v2.joblib')
            model_exists = os.path.exists(model_path)
            if model_exists:
                mobj = joblib.load(model_path)
                model = mobj['model']
                feat_cols = mobj['features']
            else:
                model = None
            # score candidates (predict probability) if model exists else use simple heuristics
            def score_row(r):
                reasons = []
                score = None
                if model is not None:
                    # build input row with expected columns
                    X = {}
                    for f in feat_cols:
                        if f in r.index:
                            X[f] = r[f]
                        else:
                            # defaults
                            if 'price' in f: X[f]=r.get('price',0)
                            else: X[f]='unknown'
                    xin = pd.DataFrame([X])
                    try:
                        score = model.predict_proba(xin)[0][1]
                    except Exception:
                        score = None
                # simple heuristics if no model
                if score is None:
                    # heuristic: higher score for mid/low price (in range), good availability pct and more amenities
                    price = float(r.get('price') if pd.notnull(r.get('price')) else 1e6)
                    pscore = 1 - (price - min_price) / max(1,(max_price - min_price))
                    amen = float(r.get('num_amenities',0))
                    avail_pct = float(r.get('pct_available',0) if 'pct_available' in r.index else 0.5)
                    score = max(0, min(1, 0.4*pscore + 0.4*(avail_pct) + 0.2*(min(amen,10)/10)))
                # generate human-friendly reasons
                if pd.notnull(r.get('num_reviews')) and r.get('num_reviews',0) > 50:
                    reasons.append('Well-reviewed listing (many reviews)')
                if pd.notnull(r.get('pct_available')) and r.get('pct_available',0) < 0.5:
                    reasons.append('High demand (low availability this year)')
                if pd.notnull(r.get('has_wifi')) and r.get('has_wifi',0)==1:
                    reasons.append('Has WiFi and amenities')
                # price reason
                if pd.notnull(r.get('price')):
                    if r.get('price') <= (min_price + (max_price-min_price)*0.33):
                        reasons.append('Competitive price for your range')
                    elif r.get('price') >= (min_price + (max_price-min_price)*0.75):
                        reasons.append('Premium-priced option')
                # final short reason join
                if not reasons:
                    reasons.append('Matches your filters and has decent availability')
                return score, '; '.join(reasons)
            # score all candidates and sort
            candidates['score_reason'] = candidates.apply(lambda row: score_row(row), axis=1)
            candidates['pred_score'] = candidates['score_reason'].apply(lambda x: x[0])
            candidates['reason_text'] = candidates['score_reason'].apply(lambda x: x[1])
            candidates = candidates.sort_values('pred_score', ascending=False)
            # show top 10
            top = candidates.head(10).copy()
            display_cols = ['id','name'] if 'name' in top.columns else ['id']
            # add price and room and neighbourhood if present
            if 'price' in top.columns: display_cols.append('price')
            if 'room_type_final' in top.columns: display_cols.append('room_type_final')
            if 'neighbourhood_final' in top.columns: display_cols.append('neighbourhood_final')
            display_cols += ['pred_score','reason_text']
            st.subheader('Top recommendations')
            st.dataframe(top[display_cols].reset_index(drop=True).head(10))
            # show each with short card
            for idx, r in top.head(5).iterrows():
                st.markdown(f\"\"\"**Listing {int(r['id'])} — {r.get('name','(no title)')}**  \n\nPrice: ${r.get('price','N/A')} — {r.get('room_type_final','')} — {r.get('neighbourhood_final','')}  \n**Why this:** {r.get('reason_text')}  \n**Predicted booking score:** {r.get('pred_score'):.3f}  \n\n---\"\"\")
st.markdown('---')
st.markdown('Notes: The booking target used here is a proxy (listing has >=1 review). For true booking availability use calendar and reservation data; this app uses calendar.csv when available to check date availability.')
