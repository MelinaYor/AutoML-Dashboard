#Importing libraries
import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg

st.title('PyCaret App')

#Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    #Column to Drop
    cols_to_drop = st.multiselect('Select columns to drop', data.columns)
    
    #Predict Column
    predict_column = st.selectbox('Select column to predict', data.columns)
    
    #Task Type Detection
    task_type = 'classification' if data[predict_column].nunique() < 20 else 'regression'
    st.write('Task Type: ', task_type)
    
    #Dealing With Null Values
    
    st.write('- Missing: Treat missing values as a separate category')
    st.write('- Most Frequent: Fill missing values with the most frequent value')
    st.write('- Mean: Fill missing values with the mean value (for numerical data)')
    st.write('- Median: Fill missing values with the median value (for numerical data)')
    st.write('- Mode: Fill missing values with the mode value')
    cat_impute_method = st.selectbox('What do you want to do with categorical data?', ['most_frequent', 'missing'])
    num_impute_method = st.selectbox('What do you want to do with numerical data?', ['mean', 'median', 'mode'])
    
    if st.button('Update Data'):
        # Preprocess data
        df = data.drop(columns=cols_to_drop)
        for col in df.columns:
            if col != predict_column:
                if df[col].dtype == 'object':
                    if cat_impute_method == 'most_frequent':
                        df[col] = df[col].fillna(df[col].mode()[0])
                    else:
                        df[col] = df[col].fillna('missing')
                else:
                    df[col] = df[col].fillna(df[col].agg(num_impute_method))
        
        #Drop rows with missing target values
        df = df.dropna(subset=[predict_column])
        
        #Save df to session state > Updated data after pre processing > We used session state, since streamlit when dealing with buttons it doesn't save for pycaret to find the updated data.
        st.session_state.df = df
        
        st.write(df)
    
    if st.button('Run PyCaret'):
        #Run PyCaret with updated data
        if 'df' in st.session_state:
            df = st.session_state.df
            if task_type == 'classification':
                clf = setup(data = df, target = predict_column, html = False)
                best_model = compare_models()
            else:
                reg = setup_reg(data = df, target = predict_column, html = False)
                best_model = compare_models_reg()
            
            st.write('Best Model: ', type(best_model).__name__)
        else:
            st.write('Please update the data first.')
