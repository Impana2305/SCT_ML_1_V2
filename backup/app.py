import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. Custom Page Configurations & Styling ---
st.set_page_config(
    page_title="House Price Prediction", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Gradient Title */
    .app-title {
        background: linear-gradient(135deg, #1E3A8A 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Modern Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(229, 231, 235, 0.6);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .valuation-label {
        color: #4B5563; 
        margin-bottom: 5px; 
        font-size: 14px; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .valuation-price {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0; 
        font-size: 3.2rem; 
        font-weight: 800;
    }
    
    .valuation-desc {
        color: #6B7280; 
        font-size: 0.9rem; 
        margin-top: 10px;
        line-height: 1.4;
    }
    
    /* Section Headers */
    .section-header {
        color: #1E3A8A;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #10B981;
        padding-left: 10px;
    }
    
    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .badge-info {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    
    /* Custom style for sidebar */
    .stSidebar {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Core Machine Learning Logic (Cached) ---
@st.cache_resource
def load_and_train_model():
    data_path = os.path.join("data", "Housing.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError()
        
    df_raw = pd.read_csv(data_path)
    features = ['area', 'bedrooms', 'bathrooms']
    
    # Preprocessing: IQR Outlier handling
    df_clean = df_raw.copy()
    for col in features:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        df_clean = df_clean[(df_clean[col] >= (q1 - 1.5 * iqr)) & (df_clean[col] <= (q3 + 1.5 * iqr))]
        
    # Feature Engineering
    df_clean['area_per_bedroom'] = df_clean['area'] / (df_clean['bedrooms'] + 0.1)
    df_raw['area_per_bedroom'] = df_raw['area'] / (df_raw['bedrooms'] + 0.1)
    
    X = df_clean[['area', 'bedrooms', 'bathrooms', 'area_per_bedroom']]
    y = df_clean['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred_test = lr_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    metrics = {
        'mae': mae,
        'r2': r2,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape
    }
    
    return lr_model, scaler, df_clean, df_raw, metrics

# --- 3. App Header ---
st.markdown('<h1 class="app-title">🏠 House Price Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">An elegant, machine-learning-driven platform built to calculate real estate market value estimations with statistical precision.</p>', unsafe_allow_html=True)

try:
    model, scaler, df_clean, df_raw, metrics = load_and_train_model()
    
    # --- 4. Sidebar Controls ---
    st.sidebar.markdown("### 📋 Property Input Controls")
    st.sidebar.write("Adjust features below to dynamically update the house valuation prediction.")
    
    input_area = st.sidebar.number_input(
        "Total Living Area (Square Feet):", 
        min_value=int(df_clean['area'].min()), 
        max_value=int(df_clean['area'].max()), 
        value=2200, 
        step=100,
        help="Enter the total interior living space of the property."
    )
    
    input_bed = st.sidebar.slider(
        "Total Bedrooms:", 
        min_value=int(df_clean['bedrooms'].min()), 
        max_value=int(df_clean['bedrooms'].max()), 
        value=3, 
        help="Select the number of bedrooms."
    )
    
    input_bath = st.sidebar.slider(
        "Total Bathrooms:", 
        min_value=int(df_clean['bathrooms'].min()), 
        max_value=int(df_clean['bathrooms'].max()), 
        value=2, 
        help="Select the number of bathrooms."
    )
    
    # Dynamic feature computation
    input_ratio = input_area / (input_bed + 0.1)
    
    # --- 5. Generate Prediction ---
    user_features = np.array([[input_area, input_bed, input_bath, input_ratio]])
    user_scaled = scaler.transform(user_features)
    predicted_val = model.predict(user_scaled)[0]
    final_val = max(0.0, predicted_val)

    # --- 6. Navigation Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "🏠 Valuation Estimator", 
        "📊 Market Trends Explorer", 
        "🧠 Model Diagnostics & Analytics"
    ])
    
    # --- TAB 1: Valuation Estimator ---
    with tab1:
        col_val_left, col_val_right = st.columns([1, 1.2])
        
        with col_val_left:
            st.markdown('<h3 class="section-header">Property Details</h3>', unsafe_allow_html=True)
            
            # Display property specs summarized
            specs_html = f"""
            <div style="background-color: #F8FAFC; border-radius: 12px; padding: 20px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
                <p style="margin: 0; font-weight: 600; color: #4B5563;">Selected Specifications:</p>
                <ul style="margin-top: 10px; margin-bottom: 0; padding-left: 20px; color: #1F2937;">
                    <li><b>Living Area:</b> {input_area:,} sq. ft.</li>
                    <li><b>Bedrooms:</b> {input_bed}</li>
                    <li><b>Bathrooms:</b> {input_bath}</li>
                    <li><b>Spatial Ratio:</b> {input_ratio:.1f} sq. ft. / bedroom</li>
                </ul>
            </div>
            """
            st.markdown(specs_html, unsafe_allow_html=True)
            
            # Beautiful prediction card
            st.markdown(f"""
                <div class="glass-card">
                    <div class="valuation-label">Estimated Market Valuation</div>
                    <h1 class="valuation-price">${final_val:,.2f}</h1>
                    <p class="valuation-desc">Valuation dynamically calculated using multivariate linear regression coefficients fit on local real estate records.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Spatial layouts analysis text
            st.markdown('<h3 class="section-header">Architectural Insights</h3>', unsafe_allow_html=True)
            if input_ratio < 400:
                st.markdown('<span class="badge badge-warning">Compact Layout</span>', unsafe_allow_html=True)
                st.info("⚠️ **Note:** The spatial layout looks quite compact relative to the total bedroom count. This may indicate smaller individual bedroom sizes.")
            elif input_ratio > 900:
                st.markdown('<span class="badge badge-success">Premium Layout</span>', unsafe_allow_html=True)
                st.success("✨ **Note:** This layout features highly premium, spacious room distributions. Ideal for higher-end luxury appeal.")
            else:
                st.markdown('<span class="badge badge-info">Balanced Layout</span>', unsafe_allow_html=True)
                st.success("✅ **Note:** This layout presents a highly balanced, comfortable space distribution.")
                
        with col_val_right:
            st.markdown('<h3 class="section-header">Live Market Position</h3>', unsafe_allow_html=True)
            
            # Interactive valuation scatter plot
            fig_val = px.scatter(
                df_clean,
                x="area",
                y="price",
                color="bedrooms",
                labels={
                    "area": "Total Area (sq. ft.)",
                    "price": "Price ($)",
                    "bedrooms": "Bedrooms"
                },
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            # Add user prediction
            fig_val.add_trace(
                go.Scatter(
                    x=[input_area],
                    y=[final_val],
                    mode="markers",
                    name="Your Property",
                    marker=dict(
                        symbol="star",
                        size=20,
                        color="#F59E0B",
                        line=dict(color="#B45309", width=2)
                    ),
                    hovertemplate="<b>Your Valuation</b><br>Area: %{x} sq. ft.<br>Value: $%{y:,.2f}<extra></extra>"
                )
            )
            fig_val.update_layout(
                legend_title_text="Bedrooms",
                hovermode="closest",
                margin=dict(l=0, r=0, t=10, b=0),
                height=450
            )
            st.plotly_chart(fig_val, use_container_width=True)
            st.caption("Each circle represents an actual home sales record. The gold star highlights where your proposed property and its predicted price fit in the market.")

    # --- TAB 2: Market Explorer ---
    with tab2:
        st.markdown('<h3 class="section-header">Housing Dataset Statistics</h3>', unsafe_allow_html=True)
        
        # Summary statistics columns
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Total Properties Analyzed", f"{len(df_clean)}")
        col_stat2.metric("Average Home Price", f"${df_clean['price'].mean():,.0f}")
        col_stat3.metric("Average Home Area", f"{df_clean['area'].mean():,.0f} sq. ft.")
        col_stat4.metric("Average Bedrooms", f"{df_clean['bedrooms'].mean():.1f}")
        
        # Price distribution chart
        st.markdown('<h3 class="section-header">Market Price Distribution</h3>', unsafe_allow_html=True)
        fig_dist = px.histogram(
            df_clean,
            x="price",
            nbins=35,
            labels={"price": "Market Price ($)", "count": "Property Count"},
            color_discrete_sequence=["#3B82F6"],
            template="plotly_white"
        )
        fig_dist.add_vline(
            x=final_val,
            line_dash="dash",
            line_color="#10B981",
            line_width=3,
            annotation_text=f"Your Property: ${final_val:,.0f}",
            annotation_position="top right"
        )
        fig_dist.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("The histogram displays the frequency of property prices in the cleaned training dataset. The green vertical line marks your predicted price.")
        
        # Area vs Price vs Bathrooms & Heatmap in columns
        col_exp_left, col_exp_right = st.columns(2)
        
        with col_exp_left:
            st.markdown('<h3 class="section-header">Area & Bathrooms Correlation</h3>', unsafe_allow_html=True)
            fig_scatter_bath = px.scatter(
                df_clean,
                x="area",
                y="price",
                color="bathrooms",
                labels={
                    "area": "Area (sq. ft.)",
                    "price": "Price ($)",
                    "bathrooms": "Bathrooms"
                },
                color_continuous_scale="Cividis",
                template="plotly_white"
            )
            fig_scatter_bath.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=350)
            st.plotly_chart(fig_scatter_bath, use_container_width=True)
            
        with col_exp_right:
            st.markdown('<h3 class="section-header">Feature Correlation Heatmap</h3>', unsafe_allow_html=True)
            corr_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
            available_corr = [c for c in corr_cols if c in df_clean.columns]
            corr_matrix = df_clean[available_corr].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels=dict(color="Correlation"),
                template="plotly_white"
            )
            fig_corr.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=350)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("A correlation value close to +1.0 indicates a strong positive linear relationship, while values near 0.0 indicate no linear relationship.")

    # --- TAB 3: Model Diagnostics ---
    with tab3:
        st.markdown('<h3 class="section-header">Machine Learning Model Info</h3>', unsafe_allow_html=True)
        st.write("This application uses a standard linear regression baseline, normalized using Scikit-Learn standard scaling.")
        
        col_diag_left, col_diag_right = st.columns(2)
        
        with col_diag_left:
            st.markdown('<h3 class="section-header">Accuracy Diagnostics</h3>', unsafe_allow_html=True)
            col_met1, col_met2 = st.columns(2)
            # Standard metrics cards
            col_met1.metric(
                label="R-squared (R² Score)",
                value=f"{metrics['r2']:.4f}",
                help="Proportion of the variance in house prices that is predictable from the input features. 1.0 is a perfect score."
            )
            col_met2.metric(
                label="Mean Absolute Error (MAE)",
                value=f"${metrics['mae']:,.2f}",
                help="The average absolute dollar amount that the model predictions deviate from the actual sales price."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            # Feature influence bar chart
            st.markdown('<h4 style="color:#1E3A8A; font-weight:600;">Feature Weights Influence</h4>', unsafe_allow_html=True)
            coefs = model.coef_
            coef_df = pd.DataFrame({
                'Feature': ['Area', 'Bedrooms', 'Bathrooms', 'Area per Bedroom'],
                'Coefficient Weight': coefs
            }).sort_values(by='Coefficient Weight', ascending=True)
            
            fig_coef = px.bar(
                coef_df,
                x='Coefficient Weight',
                y='Feature',
                orientation='h',
                color='Coefficient Weight',
                color_continuous_scale='Tealrose',
                labels={'Coefficient Weight': 'Standardized Impact Coefficient'},
                template='plotly_white'
            )
            fig_coef.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, showlegend=False)
            st.plotly_chart(fig_coef, use_container_width=True)
            st.caption("These weights demonstrate how many standard deviations the price is expected to adjust based on a standard deviation shift in the feature.")
            
        with col_diag_right:
            st.markdown('<h3 class="section-header">Actual vs. Predicted Validation</h3>', unsafe_allow_html=True)
            
            diag_df = pd.DataFrame({
                'Actual Price': metrics['y_test'],
                'Predicted Price': metrics['y_pred_test']
            })
            
            fig_diag = px.scatter(
                diag_df,
                x='Actual Price',
                y='Predicted Price',
                template='plotly_white',
                opacity=0.75
            )
            # Fit line y=x
            min_val = min(diag_df['Actual Price'].min(), diag_df['Predicted Price'].min())
            max_val = max(diag_df['Actual Price'].max(), diag_df['Predicted Price'].max())
            fig_diag.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='y = x Line',
                    line=dict(color='#EF4444', dash='dash', width=2)
                )
            )
            fig_diag.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_diag, use_container_width=True)
            st.caption("The red dashed line represents ideal predictions. Points close to the line indicate highly accurate predictions by the regression model.")

except Exception as e:
    st.error(f"❌ Error initializing dataset or model: {e}")
    st.error("Please ensure 'Housing.csv' is placed properly inside your 'data/' folder.")