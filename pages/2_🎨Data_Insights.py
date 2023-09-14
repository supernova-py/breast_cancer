import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.markdown("Developed by __supernova__: https://github.com/supernova-py")

#Differentiate vs Grade

st.header('1. Question: How does the level of tumor differentiation relate to its grade?📈')

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

#Age vs Grades: Density

st.header('2. Question: How does a woman\'s age relate to her cancer grade?⏬')

st.markdown("""
### Explore the insights below, where density represents how likely or common specific cancer grades are among women of different ages:

- **1st Grade at Density 0.005:** This indicates that women of various ages have a relatively low likelihood (density of 0.005) of having 1st-grade breast cancer. In simpler terms, 1st-grade breast cancer is less common among women in the dataset, regardless of their age.

- **2nd Grade at Density 0.025:** Women of different ages have a moderate likelihood (density of 0.025) of having 2nd-grade breast cancer. This suggests that 2nd-grade breast cancer is more common compared to 1st grade but still not highly prevalent among women in the dataset.

- **3rd Grade at Density 0.035:** Women of varying ages have a relatively higher likelihood (density of 0.035) of having 3rd-grade breast cancer. This indicates that 3rd-grade breast cancer is more common among women in the dataset compared to 1st and 2nd grades.

In summary, the density in this context helps us understand how breast cancer grades are distributed among women of different ages. It shows us the likelihood of each grade occurring, with 3rd grade being the most common, followed by 2nd grade, and 1st grade being the least common in the dataset.
""")

#age vs grade

#this also works but it's creepy-colored

#age_vs_grade_stack = sns.displot(data=df, x="Age", hue="Grade", multiple="stack", kind="kde")
#st.pyplot(age_vs_grade_stack)

custom_palette = "viridis"

age_vs_grade_stack = sns.displot(data=df, x="Age", hue="Grade", multiple="stack", kind="kde", palette=custom_palette)

plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Age vs. Grade Distribution")
st.pyplot(age_vs_grade_stack)

#Tumor Size vs T Stage

st.header('3. Question: How does tumor size relate to T stage? ⏬')
st.markdown("""
- T stage is a classification system used in cancer staging to assess tumor size and invasion into nearby tissues. Higher T stages indicate larger tumors and more extensive invasion.
    ![Tumor Stages](https://almostadoctor.co.uk/wp-content/uploads/2017/06/stages-of-bowel-cancer.png)
- ...In general, higher T stages often correlate with larger tumor sizes, indicating more advanced cancer. However, this correlation varies based on cancer type and other factors.

- Remember, T stage and tumor size aren't the same. T stage considers more factors for a comprehensive assessment of cancer extent, while tumor size focuses solely on the primary tumor's physical size.
""")

st.markdown("""
 Just a simple linear regression here... 🩺
""")

#Linear regression visualization 
# Encode the 'T Stage' column (label encoding)
label_encoder = LabelEncoder()
df['T Stage '] = label_encoder.fit_transform(df['T Stage '])

# Select the relevant columns
X = df[['T Stage ']]
y = df['Tumor Size']

# Fit
model = LinearRegression()
model.fit(X, y)

df['Predicted Tumor Size'] = model.predict(X)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='T Stage ', y='Predicted Tumor Size', palette='viridis', s=100, ax=ax)
sns.lineplot(data=df, x='T Stage ', y='Predicted Tumor Size', color='red', linewidth=2, label='Regression Line', ax=ax)

ax.set_xlabel('T Stage')
ax.set_ylabel('Predicted Tumor Size')
ax.set_title('T Stage vs. Predicted Tumor Size')

st.pyplot(fig)

#Bars visualisation
# Encode the 'T Stage ' column using label encoding

st.markdown("""
 As the T Stage increases, indicating more advanced cancer, the average predicted tumor size also tends to increase. This aligns with the medical understanding that more advanced tumors are generally larger. 🩺
""")

label_encoder = LabelEncoder()
df['T Stage '] = label_encoder.fit_transform(df['T Stage '])

X = df[['T Stage ', 'Reginol Node Positive']]
y = df['Tumor Size']  

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df, x='T Stage ', y='Tumor Size', hue='Reginol Node Positive', palette='viridis')

plt.xlabel('T Stage ')
plt.ylabel('Average Predicted Tumor Size')
plt.title('Average Predicted Tumor Size by T Stage  and Reginol Node Positive')

# Display the legend outside the box
legend = ax.legend(title='Reginol Node Positive', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=9)

st.pyplot(plt)

#"Already bored" interactive plot-builder
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
st.title('ALready bored? Play aloud with different parameters and discover intriguing correlations!')
st.markdown("""
## Hint: Discovering Fascinating Insights

- Unlock valuable insights by combining numerical parameters:
    - Age
    - Grade
    - A Stage
    - Tumor Size
    - Survival Months
    - Regional Node Positive
    - Regional Node Examined
- Enhance your analysis by utilizing categorical parameters for color coding:
    - Race
    - Marital Status
    - Differentiation
    - Estrogen Status
    - Progesterone Status
    - Survival Status
""")


df = pd.read_csv('Breast_Cancer.csv')
st.sidebar.title('Column Selection')
x_column = st.sidebar.selectbox('Select X-axis Column:', df.columns)
y_column = st.sidebar.selectbox('Select Y-axis Column:', df.columns)
color_column = st.sidebar.selectbox('Select Color Column (for color coding):', df.columns)

create_scatterplot(df, x_column, y_column, color_column, f'Scatter Plot: {x_column} vs. {y_column}')
st.write('## Data Table')
st.dataframe(df)

#EXPERIMENTAL