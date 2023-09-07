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



# Extract numerical and categorical columns
numerical_features = ['Age', 'Tumor Size']
categorical_features = ['Race']

# Define preprocessing steps for numerical and categorical columns
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for both types of features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Streamlit app
st.title('Breast Cancer Clustering')

# Sidebar for K-Means parameters
st.sidebar.title('K-Means Clustering')
n_clusters = st.sidebar.slider('Select Number of Clusters (K)', 2, 6, 3)

# Perform K-Means clustering
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=n_clusters, random_state=0))
])

df['Cluster'] = pipeline.fit_predict(df)

# Create a scatter plot of the clustered data with a legend
st.write('## Scatter Plot of Clustered Data')
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Age'], df['Tumor Size'], c=df['Cluster'], cmap='viridis', s=100)
plt.xlabel('Age')
plt.ylabel('Tumor Size')
plt.title(f'K-Means Clustering (K={n_clusters})')

# Add a legend
handles, labels = scatter.legend_elements()
legend = plt.legend(handles, labels, title='Clusters')
plt.gca().add_artist(legend)

st.pyplot(plt)

# Data Table with Cluster Labels
st.write('## Data Table with Cluster Labels')
st.dataframe(df)

#6th stage vs t stage

#n stage vs node positive

#t stage vs tumor size