import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# Set the page title and author
st.set_page_config(page_title="Breast Cancer Analysis", page_icon="✅")
st.markdown("Developed by __supernova__: [GitHub](https://github.com/supernova-py)")

# Load the breast cancer dataset
def load_data():
    return pd.read_csv("Breast_Cancer.csv")

# Section 1: Differentiate vs Grade
def section_1(df):
    st.header('1. Question: How does the level of tumor differentiation relate to its grade? 📈')
    st.markdown("""
    ### Think of tumor behavior as getting grades – like in school. We found:
    
    - Well-behaved tumors mostly get '1st Grade,' less aggressive.
    - 'Moderate' tumors are common in all grades.
    - 'Poorly behaved' tumors often get '3rd Grade,' more aggressive.
    - Some tumors are 'undecided' and can match 'anaplastic grade.'
    - So, nicer-behaved tumors usually get lower grades, while others vary. This helps us understand breast cancer.
    """)
    
    
    pivot_table = df.pivot_table(index='differentiate', columns='Grade', aggfunc='size', fill_value=0)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
    plt.title("Differentiate vs Grade")
    plt.xlabel("Grade")
    plt.ylabel("Differentiate")
    st.pyplot(fig)
    
    # Display an image
    st.image('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fowise.net%2Fus%2Fwp-content%2Fuploads%2Fsites%2F4%2F2020%2F10%2FScreenshot-2020-09-30-at-13.32.05-1024x607.png&f=1&nofb=1&ipt=4f8e408b0f5ac8021ab1f62c91af5681bd097b9fd6f1838961e39a8b131d677c&ipo=images')

# Section 2: Age vs Grades: Density
def section_2(df):
    st.header("2. Question: How does a woman's age relate to her cancer grade? ⏬")
    st.markdown("""
    ### Explore the insights below, where density represents how likely or common specific cancer grades are among women of different ages:
    
    - **1st Grade at Density 0.005:** This indicates that women of various ages have a relatively low likelihood (density of 0.005) of having 1st-grade breast cancer. In simpler terms, 1st-grade breast cancer is less common among women in the dataset, regardless of their age.
    
    - **2nd Grade at Density 0.025:** Women of different ages have a moderate likelihood (density of 0.025) of having 2nd-grade breast cancer. This suggests that 2nd-grade breast cancer is more common compared to 1st grade but still not highly prevalent among women in the dataset.
    
    - **3rd Grade at Density 0.035:** Women of varying ages have a relatively higher likelihood (density of 0.035) of having 3rd-grade breast cancer. This indicates that 3rd-grade breast cancer is more common among women in the dataset compared to 1st and 2nd grades.
    
    In summary, the density in this context helps us understand how breast cancer grades are distributed among women of different ages. It shows us the likelihood of each grade occurring, with 3rd grade being the most common, followed by 2nd grade, and 1st grade being the least common in the dataset.
    """)
    
    
    custom_palette = "viridis"
    age_vs_grade_stack = sns.displot(data=df, x="Age", hue="Grade", multiple="stack", kind="kde", palette=custom_palette)
    
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title("Age vs. Grade Distribution")
    st.pyplot(age_vs_grade_stack)

# Section 3: Tumor Size vs T Stage
def section_3(df):
    st.header('3. Question: How does tumor size relate to T stage? ⏬')
    st.markdown("""
    - T stage is a classification system used in cancer staging to assess tumor size and invasion into nearby tissues. Higher T stages indicate larger tumors and more extensive invasion.
        ![Tumor Stages](https://almostadoctor.co.uk/wp-content/uploads/2017/06/stages-of-bowel-cancer.png)
    - ...In general, higher T stages often correlate with larger tumor sizes, indicating more advanced cancer. However, this correlation varies based on cancer type and other factors.
    
    - Remember, T stage and tumor size aren't the same. T stage considers more factors for a comprehensive assessment of cancer extent, while tumor size focuses solely on the primary tumor's physical size.
    """)
    
    st.markdown("Just a simple linear regression here... 🩺")
    
    # Linear regression 
    # Encode the 'T Stage' column
    label_encoder = LabelEncoder()
    df['T Stage '] = label_encoder.fit_transform(df['T Stage '])
    
    X = df[['T Stage ']]
    y = df['Tumor Size']
    
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

# Section 4: Bars Visualization
def section_4(df):
    st.markdown("""
    As the T Stage increases, indicating more advanced cancer, the average predicted tumor size also tends to increase. This aligns with the medical understanding that more advanced tumors are generally larger. 🩺
    """)
    
    # Encode the 'T Stage ' column
    label_encoder = LabelEncoder()
    df['T Stage '] = label_encoder.fit_transform(df['T Stage '])
    
    X = df[['T Stage ', 'Reginol Node Positive']]
    y = df['Tumor Size']
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='T Stage ', y='Tumor Size', hue='Reginol Node Positive', palette='viridis')
    
    plt.xlabel('T Stage ')
    plt.ylabel('Average Predicted Tumor Size')
    plt.title('Average Predicted Tumor Size by T Stage  and Reginol Node Positive')
    
    legend = ax.legend(title='Reginol Node Positive', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=9)
    
    st.pyplot(plt)

#Interactive Plot Builder
def create_scatterplot(df, x_column, y_column, color_column, title=None):
    st.header("""
    4. Feeling bored already?
    """)
    st.markdown("""Go ahead and explore the data yourself to uncover some intriguing insights!""")

    plt.figure(figsize=(10, 6))

    df_filtered = df.dropna(subset=[x_column, y_column])

    sns.scatterplot(data=df_filtered, x=x_column, y=y_column, hue=color_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend(title=color_column)
    st.pyplot(plt)

# Define a function to create and display the survival status pie chart
def section_5(df):
    st.header("4. Survival Analysis")

    # Create a scatter plot with two clusters (Dead or Alive)
    st.subheader("Survival Analysis: Survival Months vs. Age")

    # Filter data for Dead and Alive status
    df_dead = df[df["Status"] == "Dead"]
    df_alive = df[df["Status"] == "Alive"]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_dead, x="Survival Months", y="Age", label="Dead", alpha=0.7)
    sns.scatterplot(data=df_alive, x="Survival Months", y="Age", label="Alive", alpha=0.7)
    plt.xlabel("Survival Months")
    plt.ylabel("Age")
    plt.title("Survival Analysis")
    plt.legend()
    st.pyplot(plt)

def section_6(df):
    st.header("6. Heatmap Analysis: Social Engineering")

    # Select relevant columns for the heatmap
    heatmap_data = df[['Race', 'Marital Status', 'Age']]

    # Create a heatmap
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(heatmap_data.pivot_table(index='Race', columns='Marital Status', values='Age'), cmap="YlGnBu", annot=True)
    plt.title("Heatmap: Relationship Between Race, Marital Status, and Age: Weak correlation")
    st.pyplot(plt)

#Tumor Size vs. Survival Months
def section_7(df):
    st.header("Tumor Size vs. Survival Months")
    fig, ax = plt.subplots()

    # Create scatter plot points for "Dead" and "Alive" status
    df_dead = df[df['Status'] == 'Dead']
    df_alive = df[df['Status'] == 'Alive']
    
    ax.scatter(df_dead['Tumor Size'], df_dead['Survival Months'], alpha=0.3, label='Dead', color='red')
    ax.scatter(df_alive['Tumor Size'], df_alive['Survival Months'], alpha=0.3, label='Alive', color='green')

    ax.set_xlabel('Tumor Size')
    ax.set_ylabel('Survival Months')
    
    # Add a legend
    ax.legend()

    st.pyplot(fig)


def section_8(df):
    st.header("Progesterone Status vs. Estrogen Status")

    # Create a cross-tabulation of Progesterone Status and Estrogen Status
    crosstab = pd.crosstab(df['Progesterone Status'], df['Estrogen Status'])

    # Plot a grouped bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt="d", ax=ax)

    ax.set_xlabel("Estrogen Status")
    ax.set_ylabel("Progesterone Status")
    ax.set_title("Progesterone Status vs. Estrogen Status")

    st.pyplot(fig)

def section_9(df, x_variable, y_variable):
    st.header(f"Scatterplot: {x_variable} vs. {y_variable} (Regional Node Examined)")

    # Create a scatterplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(x=x_variable, y='Regional Node Examined', data=df, jitter=True, palette='Set1')

    ax.set_xlabel(x_variable)
    ax.set_ylabel('Regional Node Examined')
    ax.set_title(f"{x_variable} vs. Regional Node Examined")

    st.pyplot(fig)

# Assuming you have loaded your DataFrame 'data' from your CSV
# You can choose 'Progesterone Status' or 'Estrogen Status' as x_variable



def main():
    df = load_data() 

    st.title("Breast Cancer Analysis")

    section_1(df)
    section_2(df)
    section_3(df)
    section_4(df)
    section_5(df)
    section_6(df)
    section_7(df)
    section_8(df)
    section_9(df, x_variable='Progesterone Status', y_variable='Regional Node Examined')

    # Sidebar for interactive plot selection
    st.sidebar.title('Paragraph X: Column Selector')
    x_column = st.sidebar.selectbox('Select X-axis Column:', df.columns)
    y_column = st.sidebar.selectbox('Select Y-axis Column:', df.columns)
    color_column = st.sidebar.selectbox('Select Color Column (for color coding):', df.columns)
    
    # Add interactive widgets
    min_x = st.sidebar.slider(f'Min {x_column}', float(df[x_column].min()), float(df[x_column].max()))
    max_x = st.sidebar.slider(f'Max {x_column}', float(df[x_column].min()), float(df[x_column].max()))
    filter_checkbox = st.sidebar.checkbox('Apply Filter')
    reset_button = st.sidebar.button('Reset Filters')

    if filter_checkbox:
        df_filtered = df[(df[x_column] >= min_x) & (df[x_column] <= max_x)]
    else:
        df_filtered = df

    create_scatterplot(df_filtered, x_column, y_column, color_column, f'Scatter Plot: {x_column} vs. {y_column}')
    
    st.write('## Data Table')
    st.dataframe(df_filtered)
    
    if reset_button:
        st.experimental_rerun()

if __name__ == '__main__':
    main()
