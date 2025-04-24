import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from auth_utils import (
    create_user, authenticate_user, validate_password,
    create_session, validate_session, delete_session,
    verify_user
)

# -------------------- CONFIG --------------------
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
DATA_PATH = "data/data.csv"
STYLE_PATH = "assets/style.css"

# -------------------- AUTH SYSTEM --------------------
def login_ui():
    st.title("Welcome")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    choice = st.radio("Choose Action", ["Login", "Sign Up"])

    if choice == "Sign Up":
        if st.button("Create Account"):
            success, message = create_user(email, password)
            if success:
                st.success("Account created!")
            else:
                st.error(message)
    elif choice == "Login":
        if st.button("Login"):
            success, message, role = authenticate_user(email, password)
            if success:
                session_id = create_session(email)
                st.session_state["session_id"] = session_id
                st.session_state["authenticated"] = True
                st.session_state["user"] = email
                st.session_state["role"] = role
                st.success(f"Welcome!")
                st.success("Login successful!")
                st.stop()
                
            else:
                st.error(message)

# -------------------- USER FUNCTIONS --------------------
def user_panel():
    st.title("User Panel")
    st.write("User management functions will go here")
    # You can expand this with actual user management functions

# -------------------- ML APP FUNCTIONS --------------------
def get_clean_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(label, 0.0, float(data[key].max()), float(data[key].mean()))
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig

def add_predictions(input_data):
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader("🔎Diagnosis Prediction")
    if prediction[0] == 0:
        st.success("🟢 Result: Benign")
    else:
        st.error("🔴 Result: Malignant")

    st.info(f"Probability (Benign): {model.predict_proba(input_array_scaled)[0][0]:.2f}")
    st.info(f"Probability (Malignant): {model.predict_proba(input_array_scaled)[0][1]:.2f}")
    st.caption("Note: This application is aimed to assist medical professionals in making breast cancer diagnosis, but should not be used as a substitute for a professional diagnosis")

# -------------------- MAIN --------------------
def main():
    st.set_page_config(page_title="🩺 Breast Cancer Diagnosis Predictor", layout="wide")

    # Session management
    if "session_id" in st.session_state:
        is_valid, email = validate_session(st.session_state["session_id"])
        if not is_valid:
            st.session_state.clear()
            st.experimental_rerun()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state.get("authenticated", False):
        login_ui()
        return

    # Load style
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH) as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # Show logout
    with st.sidebar:
        st.write(f"Logged in as: {st.session_state['user']}")
        if st.button("Logout"):
            delete_session(st.session_state["session_id"])
            st.session_state.clear()
            st.success("Logged out successfully!")
            st.stop()

    # Role-based routing
    if st.session_state["role"] == "user":
        user_panel()
    else:
        # Main App UI for medical professionals
        st.title("Breast Cancer Diagnosis")
        st.write("This app is build to help medical professionals in decision making to diagnose breast cancer in patients. It predicts using a machine learning model whether a breast mass is benign or malignant based on the cell measurements received from a breast tissue. The cell measurements are input and updated by hand using the sliders in the sidebar to receive predictions.")

        input_data = add_sidebar()

        col1, col2 = st.columns([4, 1])
        with col1:
            st.plotly_chart(get_radar_chart(input_data))
        with col2:
            add_predictions(input_data)

if __name__ == "__main__":
    main()