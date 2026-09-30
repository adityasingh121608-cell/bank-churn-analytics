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
st.markdown("**Enterprise Retention Intelligence System | European Central Bank Analytics**")

tab1, tab2, tab3, tab4 = st.tabs(["🧮 Individual Risk Calculator", "📈 Risk Distribution & Features", "⚡ What-If Scenario Simulator", "📋 Customer Database"])

with tab1:
    st.subheader("Individual Customer Risk Profiler")
    c1, c2, c3, c4 = st.columns(4)
    credit_score = c1.number_input("Credit Score", 300, 850, 650)
    age = c2.number_input("Age", 18, 100, 38)
    tenure = c3.number_input("Tenure (Years)", 0, 20, 5)
    balance = c4.number_input("Balance (€)", 0.0, 300000.0, 75000.0)
    c5, c6, c7, c8 = st.columns(4)
    num_products = c5.selectbox("Num of Products", [1, 2, 3, 4], index=0)
    has_card = c6.selectbox("Has Credit Card", [0, 1], index=1)
    is_active = c7.selectbox("Is Active Member", [0, 1], index=1)
    salary = c8.number_input("Estimated Salary (€)", 10000.0, 250000.0, 50000.0)
    c9, c10 = st.columns(2)
    geography = c9.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = c10.selectbox("Gender", ["Male", "Female"])
    bal_sal = balance / (salary + 1)
    prod_dens = num_products / (tenure + 1)
    eng_score = is_active * num_products
    age_tenure = age / (tenure + 1)
    input_data = pd.DataFrame([{
        'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
        'Age': age, 'Tenure': tenure, 'Balance': balance, 'NumOfProducts': num_products,
        'HasCrCard': has_card, 'IsActiveMember': is_active, 'EstimatedSalary': salary,
        'BalanceToSalaryRatio': bal_sal, 'ProductDensity': prod_dens,
        'EngagementScore': eng_score, 'AgeTenureRatio': age_tenure
    }])
    prob = model.predict_proba(input_data)[0][1]
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Churn Probability", f"{prob*100:.1f}%")
    if prob >= 0.5:
        m2.error("Risk Status: HIGH RISK")
        m3.warning("Recommended Action: Issue Immediate Retention Offer & Personal Advisor")
    elif prob >= 0.25:
        m2.warning("Risk Status: MEDIUM RISK")
        m3.info("Recommended Action: Cross-Sell Product & Activity Nudge")
    else:
        m2.success("Risk Status: LOW RISK")
        m3.success("Recommended Action: Standard Engagement")

with tab2:
    st.subheader("Population Churn Risk & Driver Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(df, x="Balance", color="Exited", barmode="overlay", title="Balance Distribution by Churn Status")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        fig_age = px.box(df, x="Exited", y="Age", color="Exited", title="Age Profile vs Churn Status")
        st.plotly_chart(fig_age, use_container_width=True)

with tab3:
    st.subheader("Interactive What-If Scenario Simulator")
    sim_col1, sim_col2 = st.columns(2)
    sim_products = sim_col1.slider("Simulate Num of Products", 1, 4, 1)
    sim_active = sim_col2.radio("Simulate Member Activity Status", [0, 1], index=1)
    sim_input = input_data.copy()
    sim_input['NumOfProducts'] = sim_products
    sim_input['IsActiveMember'] = sim_active
    sim_input['EngagementScore'] = sim_active * sim_products
    new_prob = model.predict_proba(sim_input)[0][1]
    diff = (new_prob - prob) * 100
    st.metric("New Churn Probability", f"{new_prob*100:.1f}%", delta=f"{diff:.1f}%", delta_color="inverse")

with tab4:
    st.subheader("Cleaned Dataset Records")
    st.dataframe(df, use_container_width=True)
