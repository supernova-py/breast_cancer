import pygwalker as pyg
import pandas as pd
import streamlit.components.v1 as components
import streamlit as st
 
# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Tableau",
    layout="wide"
)
st.title("Play around with raw data")
df = pd.read_csv("Breast_Cancer.csv")
pyg_html = pyg.walk(df, return_html=True)
components.html(pyg_html, height=1000, scrolling=True)