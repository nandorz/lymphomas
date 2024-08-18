import streamlit as st
import pandas as pd
import plotly.express as px

# Set title and markdown
st.title('Dataset: Predicting Cancer Prognosis with Machine Learning')
st.markdown(
    """
    In this script, we will:
    - Load dataset from local files
    - Show the dataset as a Pandas DataFrame
    - Provide visualizations for selected columns in the dataset
    """
)

# Default dataset URL or path
default_csv_url = 'data LNH final.csv'

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(default_csv_url, delimiter=';')

data = load_data()

# Display raw data if checkbox is selected
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Display dataset description
st.markdown(
    """
    These are the dataset for modeling analysis of DLBCL patients' prognosis.
    """
)

# Drop specific columns
data = data.drop(['no', 'cluster', 'age2', 'nlr2', 'plr2', 'lmr2'], axis=1)

# Dropdown menu for selecting column for visualization
st.subheader('Visualization of Each Variable')

list_of_variable = data.columns.tolist()

# Select variable from dropdown
variable = st.selectbox('Select variable:', list_of_variable)  

def visualize_dataset(variable):
    df_select = data[[variable, 'outcome']]
    df_select['outcome'] = df_select['outcome'].replace({1: "dead", 0: "alive"})
    # Handle categorical variables
    if variable in ['sex', 'stage', 'response', 'b_symptoms', 'ecog', 'extranodal', 'ipi', 'ldh']:
        mapping = {
            'sex': {1: "male", 0: "female"},
            'stage': {1: "Stage III or IV", 0: "Stage I or II"},
            'response': {1: "response", 0: "unresponse"},
            'b_symptoms': {1: "present", 0: "not present"},
            'ecog': {1: "good", 0: "bad"},
            'ldh': {1: "elevated", 0: "normal"}
        }
        if variable in mapping:
            df_select[variable] = df_select[variable].replace(mapping[variable])
            
        # Plot for categorical variables
        fig = px.histogram(df_select, x=variable, color="outcome", barmode='group', title=f'Histogram of {variable}')
    
    else:
        # Plot for numeric variables
        fig = px.box(df_select, x="outcome", y=variable, title=f'Box Plot of {variable}')
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

visualize_dataset(variable)
