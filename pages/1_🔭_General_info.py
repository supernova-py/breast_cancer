import pandas as pd
import pandas_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

st.title("Initializing review report... This will take 5 seconds to finish. 🕰️👀")
st.markdown("Ever been faced with raw data? It's like staring at a puzzle with no picture. But fear not – we're here to change that!")
st.markdown("Take a look at the non-visualized data. It's a maze of confusion for most of us. But wait, there's a game-changer on the left side!")

df = pd.read_csv("Breast_Cancer.csv")

vizualization_df = df.profile_report()

st_profile_report(vizualization_df)