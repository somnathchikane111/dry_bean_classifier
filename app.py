import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    """Load pre-trained model, scaler, and label encoder."""
    try:
        model = joblib.load("best_bean_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Please ensure 'best_bean_model.pkl', 'scaler.pkl', and 'label_encoder.pkl' exist in the app directory."
        )
        return None, None, None


model, scaler, label_encoder = load_artifacts()

# App Header
st.title("🫘 Dry Bean Variety Classifier")
st.markdown(
    """
Predict the dry bean variety (**BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, or SIRA**) 
based on 16 computer-vision-extracted morphological and geometric features.
"""
)

st.sidebar.header("Input Bean Morphological Features")

# Sidebar input controls organized by feature type
with st.sidebar.expander("1. Dimensions & Area", expanded=True):
    area = st.number_input("Area (A)", min_value=10000.0, max_value=300000.0, value=53048.0, step=500.0)
    convex_area = st.number_input("Convex Area (C)", min_value=10000.0, max_value=300000.0, value=53768.0, step=500.0)
    equiv_diameter = st.number_input("Equivalent Diameter (Ed)", min_value=100.0, max_value=600.0, value=253.0, step=1.0)
    perimeter = st.number_input("Perimeter (P)", min_value=400.0, max_value=2200.0, value=855.0, step=5.0)

with st.sidebar.expander("2. Axes & Aspect Ratios", expanded=True):
    major_axis = st.number_input("Major Axis Length (L)", min_value=150.0, max_value=800.0, value=320.0, step=2.0)
    minor_axis = st.number_input("Minor Axis Length (l)", min_value=100.0, max_value=500.0, value=202.0, step=2.0)
    aspect_ratio = st.number_input("Aspect Ratio (K)", min_value=1.0, max_value=3.0, value=1.58, step=0.01, format="%.4f")
    eccentricity = st.number_input("Eccentricity (Ec)", min_value=0.1, max_value=1.0, value=0.75, step=0.01, format="%.4f")

with st.sidebar.expander("3. Shape & Roundness Properties", expanded=False):
    extent = st.number_input("Extent (Ex)", min_value=0.4, max_value=0.9, value=0.75, step=0.01, format="%.4f")
    solidity = st.number_input("Solidity (S)", min_value=0.85, max_value=1.0, value=0.987, step=0.001, format="%.4f")
    roundness = st.number_input("Roundness (R)", min_value=0.4, max_value=1.0, value=0.873, step=0.005, format="%.4f")
    compactness = st.number_input("Compactness (CO)", min_value=0.5, max_value=1.0, value=0.800, step=0.005, format="%.4f")

with st.sidebar.expander("4. Advanced Shape Factors", expanded=False):
    sf1 = st.number_input("Shape Factor 1 (SF1)", min_value=0.001, max_value=0.015, value=0.00656, step=0.0001, format="%.6f")
    sf2 = st.number_input("Shape Factor 2 (SF2)", min_value=0.0001, max_value=0.005, value=0.00171, step=0.0001, format="%.6f")
    sf3 = st.number_input("Shape Factor 3 (SF3)", min_value=0.3, max_value=1.0, value=0.643, step=0.005, format="%.4f")
    sf4 = st.number_input("Shape Factor 4 (SF4)", min_value=0.9, max_value=1.0, value=0.995, step=0.0005, format="%.6f")

# Build input DataFrame
feature_names = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

input_data = pd.DataFrame([[
    area, perimeter, major_axis, minor_axis, aspect_ratio, eccentricity,
    convex_area, equiv_diameter, extent, solidity, roundness, compactness,
    sf1, sf2, sf3, sf4
]], columns=feature_names)

# Display input summary
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Submitted Morphological Profile")
    st.dataframe(input_data.T.rename(columns={0: "Input Value"}), use_container_width=True)

with col2:
    st.subheader("Classification Outcome")
    predict_btn = st.button("Classify Bean", type="primary", use_container_width=True)

    if predict_btn and model is not None and scaler is not None and label_encoder is not None:
        # Scale inputs using fitted scaler
        input_scaled = scaler.transform(input_data)
        
        # Predict class
        pred_encoded = model.predict(input_scaled)[0]
        predicted_class = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"**Predicted Variety:** `{predicted_class}`")

        # Class prediction probabilities if supported by the model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            prob_df = pd.DataFrame({
                "Bean Variety": label_encoder.classes_,
                "Confidence (%)": np.round(probs * 100, 2)
            }).sort_values(by="Confidence (%)", ascending=False).reset_index(drop=True)

            st.write("##### Probability Breakdown:")
            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(data=prob_df.set_index("Bean Variety"), use_container_width=True)
        else:
            st.info("Probability estimation is not supported by the loaded model architecture.")

st.markdown("---")
st.caption("Agri-Tech Quality Control | Dry Bean Supervised Learning Pipeline")