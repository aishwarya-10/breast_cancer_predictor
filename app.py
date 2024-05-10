# ==================================================       /     IMPORT LIBRARY    /      =================================================== #
#[model]
import pickle
import base64

#[Data Transformation]
import numpy as np
import pandas as pd

#[Dashboard]
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.stylable_container import stylable_container



# Side bar
def get_sidebar_inputs(data):
    st.sidebar.header("Cell nuclei measurements for cancer diagnosis")
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

    input_dict ={}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


def get_scaled_values(input_dict):
  
    data = pd.read_csv("Data/processed_data1.csv")

    x = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = x[key].max()
        min_val = x[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

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


def predict_cancer(input_features):
    model = pickle.load(open("Model/breast_cancer_rf_smote.pkl", "rb"))

    input_array = np.array(list(input_features.values())).reshape(1, -1)

    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.markdown("""
            <div style="background-color: green; border:2px solid #ccc; border-radius:15px; padding:2px 10px; width:80px; height: 36px; margin-bottom: 16px;">
            <p style="color:#ffffff; font-weight:bold; font-size:18px;">Benign</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color: red; border:2px solid #ccc; border-radius:15px; padding:2px 10px; width:105px; height: 36px; margin-bottom: 16px;">
            <p style="color:#ffffff; font-weight:bold; font-size:18px;">Malignant</p>
            </div>
            """, unsafe_allow_html=True)
        
    st.write("Probability of being benign: ", round(model.predict_proba(input_array)[0][1], 4))
    st.write("Probability of being malignant: ", round(model.predict_proba(input_array)[0][0], 4))


# ==================================================       /     CUSTOMIZATION    /      =================================================== #
# Streamlit Page Configuration
st.set_page_config(
    page_title = "Breast Cancer Prediction",
    page_icon= "Images/ribbon.png",
    layout = "wide",
    initial_sidebar_state= "expanded"
    )

# Title
st.title(":pink[Machine Learning-based Breast Cancer Prediction]")

# Intro
st.write("""Decision support for health professionals in biopsy procedure. This app uses machine learning models to 
         predict whether a breast mass is Benign or Malignant using cell measurements. Update measurements in the 
         sidebar to get started.""")


# ==================================================       /     MODEL    /      =================================================== #
# Get data
data = pd.read_csv("Data/processed_data1.csv")

input_features = get_sidebar_inputs(data)

# Display results
col1, col2 = st.columns([4, 2], gap= "medium")

with col1:
    chart = get_radar_chart(input_features)
    st.plotly_chart(chart)

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    with stylable_container(
        key= "results",
        css_styles= """
             {
                background-color: blue;
                color: black;
                border-radius: 20px;
                padding: 10px;
                background-image: linear-gradient(90deg, #ee9ca7 0%, #ffdde1 100%);       
            }
            """
    ):
        st.subheader("Cell Prediction:")
        predict_cancer(input_features)


# streamlit run app.py