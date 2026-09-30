import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Bank Churn Intelligence", page_icon="🏦", layout="wide")

@st.cache_data
def load_and_train():
    df = pd.read_csv("bank_churn_cleaned.csv")
    X = df.drop(columns=['CustomerId', 'Surname', 'Exited'])
    y = df['Exited']
    cat_cols = ['Geography', 'Gender']
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', num_cols), ('cat', OneHotEncoder(drop='first'), cat_cols)])
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X, y)
    return df, model

df, model = load_and_train()
st.title("🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn")
st.markdown("**Enterprise Retention Intelligence System**")
st.dataframe(df.head(10))
