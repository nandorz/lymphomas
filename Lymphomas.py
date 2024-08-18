import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

def loading_dataset():

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

def data_visualization():
    # Set title and markdown
    st.title('Data Visualization: Predicting Cancer Prognosis with Machine Learning')
    st.markdown(
        """
        The following is the visualisation of data cluster of DLBCL patients who received R-CHOP regiment
        """
    )

    # Default dataset URL or path
    default_csv_url = 'data LNH final.csv'

    # Load the dataset
    @st.cache_data
    def load_data():
        return pd.read_csv(default_csv_url, delimiter=';')

    data = load_data()

    # Drop specific columns
    data = data.drop(['no', 'cluster', 'age2', 'nlr2', 'plr2', 'lmr2'], axis=1)

    # Sample data setup (replace with actual data)
    x_feat_cv = data.drop('outcome', axis=1)
    y_pred = data['outcome']

    # Standardize data
    scaler = StandardScaler()
    x_feat_scaled = pd.DataFrame(scaler.fit_transform(x_feat_cv), columns=x_feat_cv.columns)

    # Combine scaled variables with outcome for PCA
    cv = pd.concat([x_feat_scaled, pd.DataFrame({'outcome': y_pred})], axis=1)
    cv['outcome'] = cv['outcome'].astype('category')

    X = cv.drop(columns=['outcome'])
    y = cv['outcome']

    # PCA
    @st.cache_data
    def show_pca():
        pca = PCA(n_components=4)
        components = pca.fit_transform(X)
        explained_variance_ratio = pca.explained_variance_ratio_ * 100

        # Create DataFrame for components
        components_df = pd.DataFrame(components, columns=[f'PC{num+1}' for num in range(4)])
        components_df['outcome'] = y

        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=123)
        components_df['cluster'] = kmeans.fit_predict(components_df[['PC1', 'PC2', 'PC3', 'PC4']])
        silhouette_avg = silhouette_score(components_df[['PC1', 'PC2', 'PC3', 'PC4']], components_df['cluster'])
        #st.write(f'Silhouette Score: {silhouette_avg:.4f}')

        # Labeling
        components_df['label'] = np.where(components_df['cluster'] == 1, 'Survive', 'Not Survive')

        return components_df

    components_df = show_pca()

    # Define color mapping for clusters
    color_map = {'Survive': 'rgba(0, 255, 255, 0.8)', 'Not Survive': 'rgba(255, 0, 0, 0.8)'}

    @st.cache_data
    def show_fig():
        # Plotting PCA components with clustering using Plotly
        fig = go.Figure()

        # Add 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=components_df['PC1'],
            y=components_df['PC2'],
            z=components_df['PC3'],
            mode='markers',
            marker=dict(
                size=4,
                color=components_df['label'].map(color_map).astype(str),  # Map labels to colors
                opacity=0.8,
                line=dict(
                    width=2,
                    color='black'
                )
            ),
            text=components_df['label'],
            name='Cluster'
        ))

        # Update layout
        fig.update_layout(
            title='PCA Components with K-means Clustering (3D)',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor='rgba(50, 50, 50, 0.7)',
                    gridcolor='rgba(255, 255, 255, 0.5)',
                    gridwidth=2,
                    showgrid=True,
                    zerolinecolor='white',
                    zerolinewidth=2,
                    tickfont=dict(
                        color='white'
                    )
                ),
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor='rgba(50, 50, 50, 0.7)',
                    gridcolor='rgba(255, 255, 255, 0.5)',
                    gridwidth=2,
                    showgrid=True,
                    zerolinecolor='white',
                    zerolinewidth=2,
                    tickfont=dict(
                        color='white'
                    )
                ),
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor='rgba(50, 50, 50, 0.7)',
                    gridcolor='rgba(255, 255, 255, 0.5)',
                    gridwidth=2,
                    showgrid=True,
                    zerolinecolor='white',
                    zerolinewidth=2,
                    tickfont=dict(
                        color='white'
                    )
                )
            ),
            paper_bgcolor='rgb(30, 30, 30)',
            plot_bgcolor='rgb(50, 50, 50)',
            legend=dict(
                title='Cluster Label',
                orientation='h',
                xanchor='center',
                x=0.5,
                y=-0.1,
                font=dict(
                    color='white'
                )
            )
        )

        return fig

    # Display the plot
    fig = show_fig()
    st.plotly_chart(fig)

def machine_learning():
    # Load the dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('data LNH final.csv', delimiter=';')

    # Initialize data and model
    data = load_data()

    # Load the trained model
    model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

    # Streamlit app title
    st.title('Demo: Machine Learning')

    st.header(""" 
        Fill in the patient information below and we'll predict the 2-year mortality outcome!
    """)

    # Input fields
    age = st.number_input('Age (year):', min_value=0, max_value=150, value=60, step=1)
    sex = st.radio('Gender:', ['Male', 'Female'])
    sex = 1 if sex == 'Male' else 0
    hb = st.number_input('Haemoglobin (g/dL):', min_value=0.0, value=15.0, format="%.2f")
    leu = st.number_input('Leukocyte Count (/mm^3):', min_value=0.0, value=8000.0, format="%.2f") / 1000
    neu = st.number_input('Absolute Neutrophil Count (/mm^3):', min_value=0.0, value=5000.0, format="%.2f") / 1000
    lim = st.number_input('Absolute Lymphocyte Count (/mm^3):', min_value=0.0, value=1500.0, format="%.2f") / 1000
    mon = st.number_input('Absolute Monocyte Count (/mm^3):', min_value=0.0, value=800.0, format="%.2f") / 1000

    if leu < neu + lim + mon:
        st.markdown(
            '<p style="color:red;">Warning!! Please recheck your blood count input</p>',
            unsafe_allow_html=True
        )

    plt = st.number_input('Absolute Platelet Count (/mm^3):', min_value=0.0, value=300000.0, format="%.2f") / 1000
    ldh = st.number_input('LDH level in blood (unit/L):', min_value=0.0, value=150.0, format="%.2f")
    ldh = 0 if ldh < 200 else 1  # Binary conversion

    b_symptoms = st.radio('B-Symptoms:', ['Not Present', 'Present'])
    b_symptoms = 0 if b_symptoms == 'Not Present' else 1

    extranodal = st.radio('Extranodal Involvement:', ['0 or 1', 'More Than 1'])
    extranodal = 0 if extranodal == '0 or 1' else 1

    ecog = st.radio('ECOG Performance Status:', ['Good', 'Bad'])
    ecog = 0 if ecog == 'Bad' else 1

    stage = st.radio('Ann-Arbor Stage:', ['I', 'II', 'III', 'IV'])
    stage = 0 if stage in ['I', 'II'] else 1

    response = st.radio('R-CHOP Treatment Response:', ['Not Response', 'Response'])
    response = 0 if response == 'Not Response' else 1

    # Calculate 'ipi' as needed
    ipi = sex + stage + response + ecog + ldh + extranodal

    patient_data = pd.DataFrame({
        'sex': [sex],
        'stage': [stage],
        'response': [response],
        'b_symptoms': [b_symptoms],
        'ecog': [ecog],
        'extranodal': [extranodal],
        'ldh': [ldh],
        'ipi': [ipi],
        'age': [age],
        'hb': [hb],
        'leu': [leu],
        'neu': [neu],
        'lim': [lim],
        'mon': [mon],
        'plt': [plt],
    })

    if st.button('Evaluate 2-Year Mortality'):
        prognosis = model.predict(patient_data)
        prognosis_text = "Survive" if prognosis[0] == 0 else "Not Survive"
        color = "green" if prognosis[0] == 0 else "red"
        st.markdown(
            f'<h3 style="color:{color};">Based on the data input, the 2-year mortality prediction is: {prognosis_text}</h3>',
            unsafe_allow_html=True
        )  

def main():
    # Menampilkan gambar dari data yang telah di-cache
    st.title('Lymphomas')

    # Add a selectbox to the sidebar to select the page to display
    page = st.sidebar.selectbox("Select a page", ["Dataset", "Visualization", "Machine Learning Demo"])

    if page == "Dataset":
        loading_dataset()
        
    elif page == "Visualization":
        data_visualization()
    
    elif page == "Machine Learning Demo":
        machine_learning()

if __name__ == '__main__':
    main()
