import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# --- 1. Custom Page Configurations & Styling ---
st.set_page_config(
    page_title="Real Estate AI Valuer", 
    page_icon="🏠", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS to enhance the fonts, buttons, and overall feel
st.markdown("""
    <style>
    .main {
        background-color: #fafafa;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #E5E7EB;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-top: 15px;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
        border-color: #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Core Machine Learning Logic (Cached) ---
@st.cache_resource
def load_and_train_model():
    data_path = os.path.join("data", "Housing.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError()
        
    df = pd.read_csv(data_path)
    features = ['area', 'bedrooms', 'bathrooms']
    
    # Preprocessing: IQR Outlier handling
    for col in features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= (q1 - 1.5 * iqr)) & (df[col] <= (q3 + 1.5 * iqr))]
        
    # Feature Engineering
    df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 0.1)
    
    X = df[['area', 'bedrooms', 'bathrooms', 'area_per_bedroom']]
    y = df['price']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    return lr_model, scaler

# --- 3. UI App Header ---
st.title("🏠 Automated Property Valuation Portal")
st.write("An elegant, machine-learning-driven platform built to calculate real estate market value estimations with statistical precision.")

try:
    model, scaler = load_and_train_model()
    
    # --- 4. Beautiful Input Section wrapped in a clean Container ---
    with st.container():
        st.markdown("### 📋 Property Specifications")
        
        # Area input with an info tooltip
        input_area = st.number_input(
            "Total Living Area (Square Feet):", 
            min_value=500, 
            max_value=10000, 
            value=2200, 
            step=100,
            help="Enter the total interior living space of the property."
        )
        
        # Split layout using clean columns for the room sliders
        col1, col2 = st.columns(2)
        with col1:
            input_bed = st.slider("Total Bedrooms:", min_value=1, max_value=5, value=3, help="Slide to select the number of bedrooms.")
        with col2:
            input_bath = st.slider("Total Bathrooms:", min_value=1, max_value=4, value=2, help="Slide to select the number of bathrooms.")

        # Recalculate dynamic interaction feature on the fly
        input_ratio = input_area / (input_bed + 0.1)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 5. Clean Action Call & Prediction Output ---
    if st.button("🔮 Generate Market Valuation Report"):
        user_features = np.array([[input_area, input_bed, input_bath, input_ratio]])
        user_scaled = scaler.transform(user_features)
        
        predicted_val = model.predict(user_scaled)[0]
        final_val = max(0.0, predicted_val)
        
        # Display results in a beautifully styled card box
        st.markdown(f"""
            <div class="metric-card">
                <p style="color: #4B5563; margin-bottom: 5px; font-size: 16px; font-weight: 500;">ESTIMATED MARKET VALUATION</p>
                <h1 style="color: #10B981; margin: 0; font-size: 42px;">${final_val:,.2f}</h1>
                <p style="color: #6B7280; font-size: 13px; margin-top: 10px;">Valuation calculated using localized multivariate baseline coefficients.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display structural context in an accordion expander
        with st.expander("🔍 View Architectural Insights & Layout Analysis"):
            st.write(f"📊 **Spatial Density Metric:** Your layout features **{input_ratio:.1f} sq. ft.** of liveable space per bedroom.")
            if input_ratio < 400:
                st.warning("⚠️ **Note:** The spatial layout looks quite compact relative to the total bedroom count.")
            elif input_ratio > 900:
                st.info("✨ **Note:** This layout features highly premium, spacious room distributions.")
            else:
                st.success("✅ **Note:** This layout presents a highly balanced, comfortable space distribution.")

except Exception:
    st.error("❌ Setup Error: Could not load data. Please ensure 'Housing.csv' is placed properly inside your 'data/' folder.")