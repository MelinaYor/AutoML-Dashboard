# PyCaret Streamlit App

This is a guide on how to use the PyCaret Streamlit app. The app allows you to upload a CSV file, preprocess the data, and run multiple machine learning algorithms using the PyCaret library.

## Steps to Use the App

1. **Upload a CSV file**: The app allows you to upload a CSV file which is then read into a pandas DataFrame.

2. **Select columns**: You can select columns to drop and the column to predict.

3. **Task Type Detection**: The task type (classification or regression) is determined based on the number of unique values in the prediction column.

4. **Handle Missing Values**: You can choose how to handle missing values in both categorical and numerical data. The options are:
    - Missing: Treat missing values as a separate category
    - Most Frequent: Fill missing values with the most frequent value
    - Mean: Fill missing values with the mean value (for numerical data)
    - Median: Fill missing values with the median value (for numerical data)
    - Mode: Fill missing values with the mode value

5. **Update Data**: After preprocessing the data (dropping selected columns and handling missing values), the updated DataFrame is saved to the session state.

6. **Run PyCaret**: If you click the 'Run PyCaret' button, the code sets up the PyCaret environment with the preprocessed data and runs all the available models for the determined task type (classification or regression). The best model (with the highest accuracy for classification tasks or lowest RMSE for regression tasks) is then displayed.

## Installation

Before running the app, make sure to install the necessary libraries (`streamlit` and `pycaret`). You can install them using pip:

```python
pip install streamlit pycaret
```

## Running the App

To run the app, navigate to the directory containing the app file in your terminal and type:

```bash
streamlit run app.py
```

Replace `app.py` with the name of your Streamlit app file if it's different.

## Note

Running this app might take some time as PyCaret's `compare_models` function trains and evaluates multiple models. The time taken would depend on the size and complexity of your dataset as well as the computational resources of your machine.

