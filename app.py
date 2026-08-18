import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #ddd;
    }

    .prediction-value {
        font-size: 40px;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_files():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


model, scaler = load_files()


# -----------------------------
# Feature Names
# -----------------------------
features = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT"
]


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🏠 House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Regression Model</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Input Section
# -----------------------------
st.markdown(
    '<div class="section-title">📊 Enter Property Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    CRIM = st.number_input(
        "CRIM - Crime Rate",
        min_value=0.0,
        value=3.0,
        step=0.01
    )

    ZN = st.number_input(
        "ZN - Residential Land %",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    INDUS = st.number_input(
        "INDUS - Non-Retail Business %",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

    CHAS = st.selectbox(
        "CHAS - Charles River",
        [0, 1]
    )

    NOX = st.number_input(
        "NOX - Nitric Oxide",
        min_value=0.0,
        value=0.5,
        step=0.01
    )


with col2:
    RM = st.number_input(
        "RM - Average Rooms",
        min_value=1.0,
        value=6.0,
        step=0.1
    )

    AGE = st.number_input(
        "AGE - Old Buildings %",
        min_value=0.0,
        value=60.0,
        step=0.1
    )

    DIS = st.number_input(
        "DIS - Distance to Employment",
        min_value=0.0,
        value=4.0,
        step=0.1
    )

    RAD = st.number_input(
        "RAD - Highway Accessibility",
        min_value=0.0,
        value=4.0,
        step=1.0
    )

    TAX = st.number_input(
        "TAX - Property Tax",
        min_value=0.0,
        value=300.0,
        step=1.0
    )


with col3:
    PTRATIO = st.number_input(
        "PTRATIO - Pupil Teacher Ratio",
        min_value=0.0,
        value=18.0,
        step=0.1
    )

    B = st.number_input(
        "B - Demographic Index",
        min_value=0.0,
        value=350.0,
        step=1.0
    )

    LSTAT = st.number_input(
        "LSTAT - Lower Status %",
        min_value=0.0,
        value=12.0,
        step=0.1
    )


# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame([[
    CRIM,
    ZN,
    INDUS,
    CHAS,
    NOX,
    RM,
    AGE,
    DIS,
    RAD,
    TAX,
    PTRATIO,
    B,
    LSTAT
]], columns=features)


# -----------------------------
# Feature Visualization
# -----------------------------
st.divider()

st.markdown(
    '<div class="section-title">📈 Input Feature Visualization</div>',
    unsafe_allow_html=True
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        features,
        input_data.iloc[0].values
    )

    ax.set_title("Property Feature Values")
    ax.set_xlabel("Features")
    ax.set_ylabel("Value")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


with chart_col2:

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        features,
        input_data.iloc[0].values,
        marker="o"
    )

    ax.set_title("Feature Profile")
    ax.set_xlabel("Features")
    ax.set_ylabel("Value")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


# -----------------------------
# Input Summary
# -----------------------------
st.markdown(
    '<div class="section-title">📋 Input Summary</div>',
    unsafe_allow_html=True
)

st.dataframe(
    input_data,
    use_container_width=True
)


# -----------------------------
# Prediction Button
# -----------------------------
st.divider()

predict = st.button(
    "🔮 Predict House Price"
)


# -----------------------------
# Prediction
# -----------------------------
if predict:

    try:

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(scaled_data)

        price = prediction[0]

        st.markdown(
            f"""
            <div class="prediction-box">
                <div>🏠 Predicted House Price</div>
                <div class="prediction-value">
                    ${price:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("Prediction completed successfully!")

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Built with Python • Streamlit • Scikit-learn • Joblib"
)