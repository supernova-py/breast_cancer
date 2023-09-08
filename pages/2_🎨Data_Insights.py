import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from streamlit_extras.chart_container import chart_container
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from telescopes.main_info import info
#from utils.plots import plot_galaxies
#from telescopes.references import image_quality_refs

st.markdown("Developed by __supernova__: https://github.com/supernova-py")

#Differentiate vs Grade

st.header('1. Question: How does the level of tumor differentiation relate to its grade?')

st.markdown("""
### Think of tumor behavior as getting grades – like in school. We found:

- Well-behaved tumors mostly get '1st Grade,' less aggressive.
- 'Moderate' tumors are common in all grades.
- 'Poorly behaved' tumors often get '3rd Grade,' more aggressive.
- Some tumors are 'undecided' and can match 'anaplastic grade.'
- So, nicer-behaved tumors usually get lower grades, while others vary. This helps us understand breast cancer.
""")

df = pd.read_csv("Breast_Cancer.csv")

# Create a pivot table to summarize the df
pivot_table = df.pivot_table(index='differentiate', columns='Grade', aggfunc='size', fill_value=0)

# Create a heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
plt.title("Differentiate vs Grade")
plt.xlabel("Grade")
plt.ylabel("differentiate")
st.pyplot(fig)

st.image('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fowise.net%2Fus%2Fwp-content%2Fuploads%2Fsites%2F4%2F2020%2F10%2FScreenshot-2020-09-30-at-13.32.05-1024x607.png&f=1&nofb=1&ipt=4f8e408b0f5ac8021ab1f62c91af5681bd097b9fd6f1838961e39a8b131d677c&ipo=images')

#age distribution between grades
age_vs_grade = sns.displot(data=df, x="Age", hue="Grade", multiple="stack", kind="kde")
st.pyplot(age_vs_grade)

#6th stage vs t stage

#t stage vs node positive
# Function to create a scatter plot
def create_scatterplot(df, x_column, y_column, color_column, title=None):
    plt.figure(figsize=(10, 6))

    # Filter the DataFrame to exclude rows with missing or invalid values in x_column and y_column
    df_filtered = df.dropna(subset=[x_column, y_column])

    sns.scatterplot(data=df_filtered, x=x_column, y=y_column, hue=color_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend(title=color_column)
    st.pyplot(plt)

# Streamlit app
st.title('Breast Cancer Data Visualization')

# Load your dataset from a CSV file
# Replace 'your_dataset.csv' with the actual path to your CSV file
df = pd.read_csv('Breast_Cancer.csv')

# Sidebar for selecting columns
st.sidebar.title('Column Selection')
x_column = st.sidebar.selectbox('Select X-axis Column:', df.columns)
y_column = st.sidebar.selectbox('Select Y-axis Column:', df.columns)
color_column = st.sidebar.selectbox('Select Color Column (for color coding):', df.columns)

# Plot the scatter plot using the selected columns and color column
create_scatterplot(df, x_column, y_column, color_column, f'Scatter Plot: {x_column} vs. {y_column}')

# Display the DataFrame
st.write('## Data Table')
st.dataframe(df)
#t stage vs tumor size