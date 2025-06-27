import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Hotel Booking Prediction", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("hotel_pipeline.pkl")

pipeline: Pipeline = load_pipeline()

st.title("🔍 Hotel Booking Cancellation Prediction")

st.subheader("Upload your bookings CSV file to predict cancellations:")
file = st.file_uploader("Upload a CSV file", type=["csv"])
input_df = None
if file:
    raw_df = pd.read_csv(file)
    df = raw_df.copy()
    for col in ['Agent', 'Company']:
        if col in df:
            df[col] = df[col].astype(str)
    numeric_cols = ['LeadTime', 'ArrivalDateYear', 'ArrivalDateMonth',
                    'ArrivalDateDayOfMonth', 'ArrivalDateWeekNumber',
                    'StaysInWeekendNights', 'StaysInWeekNights', 'Adults',
                    'Babies', 'PreviousCancellations', 'IsRepeatedGuest',
                    'ADR', 'PreviousBookingsNotCanceled', 'TotalOfSpecialRequests',
                    'DaysInWaitingList', 'BookingChanges']
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in ['Children']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    for col in ['Meal', 'MarketSegment', 'DistributionChannel',
                'ReservedRoomType', 'DepositType', 'CustomerType', 'HotelType']:
        if col in df:
            df[col] = df[col].astype(str)
    input_df = df
    st.success(f"📊 Dataset contains **{len(input_df)}** rows")


if input_df is not None and not input_df.empty:
    if st.button("🎯 PREDICT", type="primary"):        
        preds = pipeline.predict(input_df)
        proba = pipeline.predict_proba(input_df)[:, 1]

        results = input_df.copy()
        results['Prediction'] = np.where(preds==1, 'CANCELLED', 'HONORED')
        results['Probability'] = proba
        results['Risk'] = pd.cut(proba, bins=[0,0.3,0.7,1.0], labels=['Low','Medium','High'])

        st.subheader(f"📈 Batch Results ({len(input_df)})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", len(input_df))
        c2.metric("Predicted Cancellations", int((results['Prediction']=='CANCELLED').sum()))
        c3.metric("Cancellation Rate", f"{(results['Prediction']=='CANCELLED').mean():.1%}")
        c4.metric("Avg Probability", f"{results['Probability'].mean():.1%}")

        st.dataframe(results[['Prediction','Probability','Risk']], use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.hist(proba, bins=20, edgecolor='black')
            ax.set_xlabel('Probability of Cancellation')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            plt.close()


