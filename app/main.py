import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# function to fetch the cleaned data of the model (we need clean data for the sidebar component which has the independent variables)
def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


# sidebar component for input data(cell measurements)
def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  
  data = get_clean_data()
  
  # sliding function for the independent variables (columns) each with a label
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

# input dictionary key function used to store the input measurements so as to create the chart and the prediction 
  input_dict = {}

# loop each label for their values
# the key select the column in the data associated to an independent variable
# the two values in the above slider_lables list consists of a label and a key consecutively

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label, # first value in slider_lables
      min_value=float(0), # minimum value of a label
      max_value=float(data[key].max()), # maximum value of a label
      value=float(data[key].mean())
    )
    
  return input_dict


def main():
  st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
input_data = add_sidebar()
st.write(input_data)

with st.container():
    st.title("Breast Cancer Predictor")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
  
col1, col2 = st.columns([4,1]) # creating the columns(first column(chart column) should be 4 times larger than the second column(diagnosis part))

  
 



if __name__ == '__main__':
  main()