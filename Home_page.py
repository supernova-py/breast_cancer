import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.image as mpimg
from PIL import Image


st.set_page_config(
    page_title="Home page",
    page_icon="👋",
    layout="centered")

#image = Image.open('microscope_1.png')
#st.image(image)

# Main Description
st.markdown("## 👋 Welcome to BreastCancerViz, your ultimate tool to explore and understand breast cancer datasets!")
st.markdown("Developed by __supernova__: https://github.com/supernova-py")
st.markdown("The app is still under development. Please reach me in the github repo if you have any comments or suggestions.")

# Description of the features. 
st.markdown(
    """
    ### Select on the left panel what you want to explore:

    - With 🔭 General info, Get a brief overview of the breast cancer datasets, including patient demographics and basic tumor information.
    
    - With 🎨 Data Insights, Explore genetic markers and mutations associated with breast cancer susceptibility.

    - With 📈 Prediction_Model, Dive into trends over time, examining diagnosis rates, treatment efficacy, and survival outcomes.
    
    \n  
    
    More information can be found by clicking in the README.
    """
)