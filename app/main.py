from altair import Data, DataFormat, Datasets
from sklearn import datasets
import streamlit as st # streamlit application used to create- integrates the model(backend) with the frontend
import pickle # for perfomance 
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import altair as alt


# function to fetch the cleaned data of the model

def get_clean_data():

  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


# sidebar component for input data(cell measurements)
def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")

  data = get_clean_data()

# sliding function for the independent variables(labels)
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

for label , key in slider_labels:
  st.sidebar.slider(
    label, 
    min_value=float(0),
    max_value=float([key].max())
  )


def main(): 
  st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor Application",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

add_sidebar()

with st.container(): #container for a component
     st.title("Breast Cancer Diagnosis Predictor") #title 
     st.write("This application predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.") # (st.write) for writing paragaraphs


col1, col2 = st.columns([4,1]) # creating the columns(first column(chart column) should be 4 times larger than the second column(diagnosis part))

















if __name__ == '__main__':
  main()