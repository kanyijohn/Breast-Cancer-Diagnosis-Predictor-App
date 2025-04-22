import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
import os
import bcrypt
import re  # Email validation

# -------------------- CONFIG --------------------
USER_DB = "users.json"
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
DATA_PATH = "data/data.csv"
STYLE_PATH = "assets/style.css"


# -------------------- AUTH SYSTEM --------------------
def save_user(email, password):
    users = {}
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = {}

    if email in users:
        st.warning("This email is already registered.")
        return False

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        st.warning("Invalid email format.")
        return False

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[email] = hashed

    with open(USER_DB, "w") as f:
        json.dump(users, f)
    return True


def authenticate_user(email, password):
    if not os.path.exists(USER_DB):
        return False
    with open(USER_DB, "r") as f:
        try:
            users = json.load(f)
        except json.JSONDecodeError:
            return False
    stored_hashed = users.get(email)
    if not stored_hashed:
        return False
    return bcrypt.checkpw(password.encode(), stored_hashed.encode())



def login_ui():
    st.title("🔐 Login or Sign Up")
    choice = st.radio("Choose Action", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.button("Create Account"):
            if save_user(email, password):
                st.success("Account created! Please log in.")
    elif choice == "Login":
        if st.button("Login"):
            if authenticate_user(email, password):
                st.session_state["authenticated"] = True
                st.session_state["user"] = email
                st.success(f"Welcome {email}!")
                st.success("Login successful! Please refresh the page manually.")
                st.stop()
            else:
                st.error("Invalid email or password")


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
                  'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
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
        name='Mean Values'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig


def add_predictions(input_data):
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader("🔎 Diagnosis Prediction")
    if prediction[0] == 0:
        st.success("🟢 Result: Benign")
    else:
        st.error("🔴 Result: Malignant")

    st.info(f"Probability (Benign): {model.predict_proba(input_array_scaled)[0][0]:.2f}")
    st.info(f"Probability (Malignant): {model.predict_proba(input_array_scaled)[0][1]:.2f}")
    st.caption("Note: This app is for educational purposes and should not replace professional diagnosis.")


# -------------------- MAIN --------------------
def main():
    st.set_page_config(page_title="🩺 Breast Cancer Diagnosis Predictor", layout="wide")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_ui()
        return

    # Show logout
    with st.sidebar:
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.success("Logged out successfully! Please refresh the page manually.")
            st.stop()


    # Load style
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Main App UI
    st.title("🩺 Breast Cancer Diagnosis Predictor App")
    st.write("Use the sidebar to input cell measurements and receive predictions using a trained ML model.")

    input_data = add_sidebar()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(get_radar_chart(input_data))
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
