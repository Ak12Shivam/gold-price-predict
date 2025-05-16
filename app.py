import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Generate synthetic data and train a simple DecisionTreeRegressor model
np.random.seed(42)
n_samples = 1000
synthetic_data = pd.DataFrame({
    'SPX': np.random.uniform(3000, 6000, n_samples),
    'USO': np.random.uniform(50, 100, n_samples),
    'SLV': np.random.uniform(15, 35, n_samples),
    'EUR_USD': np.random.uniform(0.9, 1.3, n_samples)
})
# Simulate gold price as a function of features with some noise
synthetic_data['Gold_Price'] = (
    0.5 * synthetic_data['SPX'] * 0.01 +
    2.0 * synthetic_data['USO'] +
    3.0 * synthetic_data['SLV'] +
    1000 * synthetic_data['EUR_USD'] +
    np.random.normal(0, 50, n_samples)
)

# Prepare features and target
X = synthetic_data[['SPX', 'USO', 'SLV', 'EUR_USD']]
y = synthetic_data['Gold_Price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_scaled, y)

# Set page configuration
st.set_page_config(
    page_title="Gold Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css');

    .main {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d97706;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
        border: 2px solid #3b82f6;
        padding: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Main container
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">Gold Price Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Input form
    st.subheader("Enter Market Data")
    col1, col2 = st.columns(2)
    
    with col1:
        spx = st.number_input("S&P 500 Index (SPX)", min_value=0.0, value=5000.0, step=0.1)
        uso = st.number_input("United States Oil Fund (USO)", min_value=0.0, value=70.0, step=0.1)
    
    with col2:
        slv = st.number_input("Silver Price (SLV)", min_value=0.0, value=25.0, step=0.1)
        eur_usd = st.number_input("EUR/USD Exchange Rate", min_value=0.0, value=1.1, step=0.01)

    # Predict button
    if st.button("Predict Gold Price"):
        try:
            # Prepare features for prediction
            features = np.array([[spx, uso, slv, eur_usd]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            
            # Store prediction
            st.session_state.predictions.append({
                "timestamp": datetime.now(),
                "SPX": spx,
                "USO": uso,
                "SLV": slv,
                "EUR_USD": eur_usd,
                "Prediction": prediction
            })
            
            # Display prediction
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Predicted Gold Price", f"${prediction:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Visualization section
    if st.session_state.predictions:
        st.subheader("Prediction History")
        df = pd.DataFrame(st.session_state.predictions)
        
        # Line chart for predictions over time
        fig = px.line(
            df,
            x="timestamp",
            y="Prediction",
            title="Gold Price Predictions Over Time",
            markers=True,
            line_shape="spline"
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Predicted Gold Price ($)",
            plot_bgcolor="rgba(255,255,255,0.8)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlations")
        corr = df[["SPX", "USO", "SLV", "EUR_USD", "Prediction"]].corr()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1
        ))
        fig_heatmap.update_layout(
            title="Correlation Matrix of Features and Predictions",
            plot_bgcolor="rgba(255,255,255,0.8)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Footer
    st.markdown("""
    <div style='text-align: center; color: white; margin-top: 2rem;'>
        Powered by Your Work Innovations | © 2025 Gold Price Predictor
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.header("About")
    st.write("""
    This dashboard predicts gold prices based on key market indicators:
    - S&P 500 Index (SPX)
    - United States Oil Fund (USO)
    - Silver Price (SLV)
    - EUR/USD Exchange Rate
    
    Enter the values and click 'Predict Gold Price' to get a prediction.
    The dashboard also shows historical predictions and feature correlations.
    Note: This uses a synthetic pre-trained model for demonstration.
    """)