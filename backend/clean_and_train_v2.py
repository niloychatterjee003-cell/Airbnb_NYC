\
#!/usr/bin/env python3
\"\"\"clean_and_train_v2.py
- Loads data from ../data (listings.csv, calendar.csv, reviews.csv)
- Cleans listings, engineers features, merges simple calendar availability
- Trains RandomForestClassifier to predict 'booked_proxy' (has >=1 review)
- Saves cleaned merged CSV and model to ../outputs/
\"\"\"
import os, joblib, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

def safe_read(path):
    if os.path.exists(path):
        print("Loading", path)
        return pd.read_csv(path, low_memory=False)
    return None

def normalize_price(df, col_candidates=None):
    if col_candidates is None:
        col_candidates = [c for c in df.columns if "price" in c.lower()]
    for c in col_candidates:
        if c in df.columns:
            s = df[c].astype(str).str.replace('[\\$,]', '', regex=True)
            return pd.to_numeric(s, errors='coerce')
    return pd.Series([np.nan]*len(df))

def basic_clean_listings(listings):
    listings = listings.copy()
    listings.columns = [c.strip() for c in listings.columns]
    listings['price'] = normalize_price(listings)
    # neighbourhood
    if 'neighbourhood_cleansed' in listings.columns:
        listings['neighbourhood_final'] = listings['neighbourhood_cleansed']
    else:
        neigh = next((c for c in listings.columns if 'neigh' in c.lower()), None)
        listings['neighbourhood_final'] = listings[neigh] if neigh and neigh in listings.columns else 'Unknown'
    # room type
    listings['room_type_final'] = listings['room_type'] if 'room_type' in listings.columns else (listings[next((c for c in listings.columns if 'room' in c.lower() and 'type' in c.lower()), None)] if any('room' in c.lower() and 'type' in c.lower() for c in listings.columns) else 'Unknown')
    # amenities features
    if 'amenities' in listings.columns:
        listings['num_amenities'] = listings['amenities'].astype(str).apply(lambda x: x.count(',')+1 if isinstance(x,str) and x.strip()!='' else 0)
        listings['has_wifi'] = listings['amenities'].astype(str).str.contains('wifi|internet', case=False, na=False).astype(int)
    else:
        listings['num_amenities'] = 0
        listings['has_wifi'] = 0
    # reviews and availability
    listings['num_reviews'] = pd.to_numeric(listings['number_of_reviews'], errors='coerce') if 'number_of_reviews' in listings.columns else pd.to_numeric(listings.get('num_reviews', pd.Series([np.nan]*len(listings))), errors='coerce')
    listings['reviews_per_month'] = pd.to_numeric(listings.get('reviews_per_month', pd.Series([np.nan]*len(listings))), errors='coerce')
    listings['availability_365'] = pd.to_numeric(listings.get('availability_365', pd.Series([np.nan]*len(listings))), errors='coerce')
    listings['host_listings_count'] = pd.to_numeric(listings.get('host_listings_count', pd.Series([0]*len(listings))), errors='coerce').fillna(0)
    # proxy target
    listings['booked_proxy'] = (listings['num_reviews'].fillna(0) > 0).astype(int)
    return listings

def calendar_availability_summary(calendar):
    # calendar has fields: listing_id, date, available ('t' or 'f') or 'available' in some variants
    if calendar is None:
        return None
    cal = calendar.copy()
    # normalize column names
    cal.columns = [c.strip() for c in cal.columns]
    # ensure listing_id and date and available exist
    id_col = next((c for c in cal.columns if 'listing' in c.lower()), None)
    date_col = next((c for c in cal.columns if 'date' in c.lower()), None)
    avail_col = next((c for c in cal.columns if 'avail' in c.lower()), None)
    if id_col is None or date_col is None or avail_col is None:
        return None
    # compute percent available across the year (simple metric)
    cal['available_flag'] = cal[avail_col].astype(str).str.lower().map(lambda x: 1 if x in ['t','true','y','yes','1','available'] else 0)
    summary = cal.groupby(id_col)['available_flag'].mean().reset_index().rename(columns={'available_flag':'pct_available'})
    summary.columns = [id_col, 'pct_available']
    return summary

def merge_calendar_into_listings(listings, calendar):
    if calendar is None:
        listings['pct_available'] = np.nan
        return listings
    summary = calendar_availability_summary(calendar)
    if summary is None:
        listings['pct_available'] = np.nan
        return listings
    id_col = summary.columns[0]
    # ensure id column name alignment
    if 'id' in listings.columns and id_col != 'id':
        summary = summary.rename(columns={id_col:'id'})
    listings = listings.merge(summary, how='left', on='id')
    listings['pct_available'] = listings.get('pct_available', np.nan)
    return listings

def train_model(listings):
    df = listings.copy()
    df = df[df['price'].notnull()]
    features = ['price','room_type_final','neighbourhood_final','availability_365','num_amenities','has_wifi','host_listings_count','is_instant_bookable','pct_available']
    features = [f for f in features if f in df.columns]
    df_model = df[features + ['booked_proxy']].copy()
    # fill numeric
    for c in df_model.select_dtypes(include=[np.number]).columns:
        df_model[c] = df_model[c].fillna(df_model[c].median() if not df_model[c].dropna().empty else 0)
    for c in df_model.select_dtypes(include=['object']).columns:
        df_model[c] = df_model[c].fillna('unknown').astype(str)
    X = df_model.drop(columns=['booked_proxy'])
    y = df_model['booked_proxy']
    if y.nunique() < 2 or len(y) < 200:
        print('Not enough variation to train. Saved cleaned file only.')
        df.to_csv(os.path.join(OUT, 'cleaned_listings_v2.csv'), index=False)
        return None
    numeric = X.select_dtypes(include=['number']).columns.tolist()
    categorical = X.select_dtypes(include=['object']).columns.tolist()
    pre = ColumnTransformer(transformers=[('num', StandardScaler(), numeric), ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical)])
    clf = Pipeline([('pre', pre), ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    print('Training with', len(X), 'rows and features:', numeric + categorical)
    clf.fit(X, y)
    joblib.dump({'model': clf, 'features': X.columns.tolist()}, os.path.join(OUT, 'rf_booked_proxy_v2.joblib'))
    df.to_csv(os.path.join(OUT, 'cleaned_listings_v2.csv'), index=False)
    print('Saved model and cleaned file to', OUT)
    return clf

def main():
    listings = safe_read(os.path.join(DATA, 'listings.csv')) or safe_read(os.path.join(DATA, 'listings.csv.gz'))
    calendar = safe_read(os.path.join(DATA, 'calendar.csv'))
    reviews = safe_read(os.path.join(DATA, 'reviews.csv'))
    if listings is None:
        raise FileNotFoundError('listings.csv not found in data/. Run fetch_insideairbnb_nyc.py first.')
    listings = basic_clean_listings(listings)
    listings = listings.rename(columns={'id':'id'})
    listings = listings.assign(is_instant_bookable=listings.get('instant_bookable', listings.get('instant_bookable', 'unknown')))
    listings = merge_calendar_into_listings(listings, calendar)
    model = train_model(listings)
    print('Done. Outputs in', OUT)

if __name__ == '__main__':
    main()
