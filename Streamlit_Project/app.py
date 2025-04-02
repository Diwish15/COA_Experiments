import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Indian housing dataset
df = pd.read_csv("indian_house_prices.csv")  # Ensure the dataset is in the same directory
df['Price (in Lakhs)'] = df['Price'] / 100000  # Convert to INR Lakhs

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Convert categorical columns using Pandas Categorical (to handle unseen labels)
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Split data
X = df.drop(columns=['Price', 'Price (in Lakhs)'])
y = df['Price (in Lakhs)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert categorical variables into numerical values
X_train = X_train.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)
X_test = X_test.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name.replace(" ", "_").lower()}.pkl')

# Streamlit App UI
st.set_page_config(page_title="Indian House Price Predictor", page_icon="🏡", layout="wide")
st.sidebar.title("🏡 Indian House Price Prediction App")

# Sidebar Navigation
page = st.sidebar.radio("Go to", ["Dashboard", "Model", "Visualization", "SHAP Analysis"])

if page == "Dashboard":
    st.title("📊 Indian House Price Dashboard")
    st.markdown("### Overview of Indian House Price Data")
    
    # Summary Statistics
    st.write("#### Dataset Summary")
    st.write(df.describe())
    
    # Price Distribution
    fig = px.histogram(df, x='Price (in Lakhs)', nbins=50, title="House Price Distribution (INR Lakhs)", color_discrete_sequence=["#3498DB"])
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model":
    st.title("🏡 Indian House Price Prediction")
    st.markdown("### Enter property details to estimate the price (in INR Lakhs or Crores)")
    
    # Sidebar for user input
    st.sidebar.header("Property Features")
    inputs = []
    for feature in X.columns:
        if feature in categorical_cols:
            # Get unique categories from dataframe
            options = list(df[feature].cat.categories)
            selected_option = st.sidebar.selectbox(f"{feature}", options)
            
            # Convert category to numeric index
            feature_value = df[feature].cat.categories.get_loc(selected_option)
            inputs.append(feature_value)
        else:
            value = st.sidebar.number_input(f"{feature}", value=float(df[feature].mean()))
            inputs.append(value)
    
    # Model Selection
    model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))
    
    # Predict button
    if st.sidebar.button("🔍 Predict Price", use_container_width=True):
        model = joblib.load(f'{model_choice.replace(" ", "_").lower()}.pkl')
        prediction = model.predict([inputs])[0]
        
        # Convert to Lakhs or Crores
        if prediction >= 100:
            price_str = f"₹{prediction / 100:.2f} Crores"
        else:
            price_str = f"₹{prediction:.2f} Lakhs"
        
        st.sidebar.success(f"💰 Estimated Price: **{price_str}**")

elif page == "Visualization":
    st.title("📊 Model Performance Visualization")
    
    model_choice = st.selectbox("Choose a Model", list(models.keys()))
    selected_model = joblib.load(f'{model_choice.replace(" ", "_").lower()}.pkl')
    y_pred = selected_model.predict(X_test)
    
    # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    st.write(f"**Model:** {model_choice}")
    st.write(f"**RMSE:** {rmse:.2f} Lakhs")  # RMSE in INR Lakhs
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Prices (Lakhs)")
    ax.set_ylabel("Predicted Prices (Lakhs)")
    ax.set_title("Actual vs Predicted House Prices")
    st.pyplot(fig)

elif page == "SHAP Analysis":
    st.title("🔍 SHAP Analysis for Feature Importance")
    
    model_choice = st.selectbox("Choose a Model for SHAP Analysis", list(models.keys()))
    selected_model = joblib.load(f'{model_choice.replace(" ", "_").lower()}.pkl')
    
    # SHAP Analysis
    explainer = shap.Explainer(selected_model, X_train)
    shap_values = explainer(X_test)

    # SHAP Summary Plot
    st.write("### SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

st.markdown("---")
st.caption("🚀 Powered by Machine Learning & Streamlit | Built for Indian Market!")
