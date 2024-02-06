# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import io
import base64
import matplotlib.pyplot as plt
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objects as go
from io import BytesIO


import streamlit.components.v1 as components
st.set_page_config(
    page_title="Data Cleaning and Visuallisation app",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="auto",  # or ""
    )
with open("J:\data science\VS code\data cleaning\style.css") as f:
    css=f.read()
st.markdown("<style>" + css +"</style>",unsafe_allow_html=True)
# Add viewport meta tag for custom HTML
components.html("<meta name='viewport' content='width=device-width, initial-scale=1.0'>", height=0)

# from imblearn.under_sampling import NearMiss
# from imblearn.over_sampling import RandomOverSampler
#******************************************************************************************************************************
# Define a session state to store variables across sessions
class SessionState:
    def __init__(self):
        self.selected_dataset = None

# Create an instance of the SessionState class
session_state = SessionState()
# Apply the CSS class to your markdown elements
st.markdown('<div class="sub-header">A Data Science Project by Akash Patil</div>', unsafe_allow_html=True)
st.markdown('''[LinkedIn](https://www.linkedin.com/in/akash-patil-985a7a179/)
[GitHub](https://github.com/akashpatil108)
''')

# Clear and appealing header with HTML/CSS styling
st.markdown(
    """
    <style>
        
        }

        
    </style>
    <div class="header">Data Analysis and Visualization</div>
    <div class="tagline">Empower Your Data Journey: Unleash Insights, Master Data Cleaning </div>
    <div class="intro">
        Welcome to the Data Analysis and Visualization App! This application allows you to perform various data cleaning operations and visualize your dataset.
    </div>
        <div class="learn">
        Dive into the Data Analysis and Visualization App, your interactive playground for uncovering insights and mastering data science concepts. This app isn't just about crunching numbers – it's about empowering you to transform raw data into clean data. Start exploring, analyzing, and learning by doing – your data awaits!

    </div>
    """,
    unsafe_allow_html=True
)

# Introduction and instructions for data loading
st.write("""
This application allows you to perform various data cleaning operations and visualize your dataset.
<details>
<summary><strong><div class="info"; title="How to use this web site">How to Use:</div><strong></summary>

1. **Load a Dataset:**
   - You can upload your custom CSV file using the 'Upload a CSV file' from sidebar.
   - OR Choose a preloaded dataset from the available options.

2. **Data Cleaning Operations:**
   - Select data cleaning operations from the sidebar to clean and preprocess your dataset.

3. **Visualizations:**
   - Choose the visualizations you want to generate for your  dataset.

4. **Download Cleaned Data:**
   - After data cleaning, you can download the cleaned dataset using the 'Download Cleaned Data' button.

Enjoy exploring and analyzing your data!
</details>
""",  unsafe_allow_html=True)

st.markdown("""
<details>
<summary><strong><div class="info"><span title='Click here to get the information'><h3>Data Cleaning</h3></span></div></strong></summary>

**What is Data Cleaning?**
Data cleaning is the process of identifying and correcting errors or inconsistencies in datasets. It involves handling missing data, removing duplicates, and ensuring data is suitable for analysis.

**Why Use Data Cleaning?**
- **Improves Accuracy:** Clean data leads to more accurate and reliable results.
- **Enhances Model Performance:** Quality data is crucial for training accurate machine learning models.
- **Ensures Consistency:** Data cleaning ensures uniformity and consistency across the dataset.


**How to Use Data Cleaning**

1. **Handle Missing Values:** Identify and fill or remove missing data points.
2. **Remove Duplicates:** Identify and eliminate identical records.
3. **Convert Data Types:** Ensure data is in the correct format for analysis.
4. **Handle Categorical Variables:** Encode or transform categorical variables for numerical analysis.
5. **Handle Datetime Variables:** Manage and manipulate date and time information.
6. **Handle Outliers:** Detect and address outliers to prevent skewed analysis.


**When to Use Data Cleaning:**

Data cleaning is an initial step in the data preprocessing phase. It is performed before exploratory data analysis (EDA) or building machine learning models. Whenever raw data is obtained, data cleaning is necessary to prepare it for analysis.

**Why We Are Using Data Cleaning:**
- **Improves Accuracy:** Clean data leads to more accurate and reliable results.
- **Enhances Model Performance:** Quality data is crucial for training accurate machine learning models.
- **Ensures Consistency:** Data cleaning ensures uniformity and consistency across the dataset.



<details>
<summary><strong><div style="text-align: center;"><h3>Data Visualization</h3></div></strong></summary>

**What is Data Visualization?**
Data visualization is the representation of data in graphical or visual format. It helps in understanding complex patterns, trends, and insights that may not be apparent in raw, textual data. Visualization is crucial for conveying information effectively.

**Where to Use Data Visualization?**
Data visualization is used in various fields, including business, science, and academia, to communicate trends, patterns, and insights. It is particularly useful when dealing with large datasets or complex relationships within the data.



**How to Use Data Visualization**

1. **Selecting Visualization Types:** Choose appropriate charts, graphs, or plots for the type of data and the insights you want to convey.
2. **Customizing Visualizations:** Adjust colors, labels, and other elements to enhance clarity.
3. **Interactivity:** Add interactive elements for a more engaging user experience.
4. **Exploratory Data Analysis (EDA):** Use visualizations to explore the dataset before in-depth analysis.
5. **Storytelling with Data:** Create a narrative through visualizations to tell a compelling story.


**When to Use Data Visualization:**
Data visualization is used throughout the data analysis process:
- **Exploratory Phase:** Understand the structure and patterns in the data.
- **Presentation Phase:** Communicate findings to stakeholders.
- **Validation Phase:** Confirm or challenge hypotheses visually.

**Why We Are Using Data Visualization:**
- **Enhances Understanding:** Visual representation aids in understanding complex data.
- **Facilitates Communication:** Conveys insights to both technical and non-technical audiences.
- **Identifies Patterns:** Visualizations make it easier to spot trends, outliers, and patterns.

</details>

</details>

""",  unsafe_allow_html=True)


#******************************************************************************************************************************
st.write("---")
# Step 4: Data Cleaning Operations

def key_generator():
    current_number = 1
    while current_number <= 1000:
        yield current_number
        current_number += 1
# Example of how to use the generator
key = key_generator()
# Function to convert data types
def convert_data_types(df):
        st.markdown(
     """
        <details>
        <summary><strong><h3><div class="info"> Convert Data Types:</div></h3></strong></summary>

        <strong>When to Use:</strong>
        - When loading a dataset, data types may be assigned incorrectly.
        - Before performing operations that require specific data types.

        <strong>Why We Are Using:</strong>
        - Ensures that each column has the appropriate data type for analysis.
        - Helps in optimizing memory usage.

        <strong>How to Use:</strong>
        1. Check the current data types of columns using `df.dtypes`.
        2. Identify columns that need to be converted to a different data type.
        3. Call this function and provide the DataFrame as an argument.

        Example:
        ```python
        # Check data types before conversion
        print("Before Conversion:")
        print(df.dtypes)
        # Convert data types
        df = convert_data_types(df)
        # Check data types after conversion
        print("After Conversion:")
        print(df.dtypes)

        ```
        <strong>Note:</strong>
        - Be cautious when converting data types to avoid loss of information.
        - This function will attempt to convert columns to the most appropriate data type.
        :param df: The input DataFrame.
        :return: DataFrame with converted data types.

        </details>
        """, unsafe_allow_html=True)
        key_value = str(next(key))
        # Option to choose columns and target data type
        columns_to_convert = st.multiselect("Select columns to convert:", df.columns,key=key_value)
        # Updated list of target data types
        target_data_types = ["int", "float", "object", "bool", "datetime64[ns]", "category"]
        key_value = str(next(key))
        target_data_type = st.selectbox("Select target data type:", target_data_types,key=key_value)
        df_cleaned = df.copy()
        if columns_to_convert and target_data_type:
            # Convert selected columns to the target data type
            df_cleaned[columns_to_convert] = df_cleaned[columns_to_convert].astype(target_data_type)
            st.success("Data types converted successfully!")
        return df_cleaned

# Function to remove duplicates
def remove_duplicates(df):
    st.markdown(
     """
    <details>
    <summary><strong><h3><div class="info">Remove Duplicates:</div></h3></strong></summary>

    <strong>What is Removing Duplicates:</strong>
    - Removing duplicates involves eliminating identical rows from a dataset.

    <strong>Where to Use Removing Duplicates:</strong>
    - Apply when datasets may contain repeated or redundant information.
    - Essential for maintaining data integrity.

    <strong>When to Use Removing Duplicates:</strong>
    - Before analysis or modeling to ensure the dataset reflects unique observations.

    <strong>Why We Are Using Removing Duplicates:</strong>
    - Duplicates can lead to biased analyses and models.
    - Removal ensures accurate insights by working with distinct data points.

    <strong>How to Use Removing Duplicates:</strong>
    1. Specify the subset of columns to consider for identifying duplicates (default is all columns).
    2. Choose the strategy for keeping rows ('first', 'last', or 'False' to remove all duplicates).
    3. Call this function with the DataFrame, subset, and keep strategy.

    Example:
    ```python
    # Identify duplicates before removal
    print("Before Removing Duplicates:")
    print(df.duplicated().sum())
    # Remove duplicates keeping the first occurrence
    df = remove_duplicates(df, subset=['column1', 'column2'], keep='first')
    # Identify duplicates after removal
    print("After Removing Duplicates:")
    print(df.duplicated().sum())
    ```

    <strong>Note:</strong>
    - 'subset' specifies columns to consider for duplicate identification.
    - 'keep' determines which duplicates to keep ('first', 'last', or 'False' for removing all).
    :param df: The input DataFrame.
    :param subset: Columns to consider for identifying duplicates, default is None (all columns).
    :param keep: Strategy for keeping duplicates ('first', 'last', or 'False' for removing all), default is 'first'.
    :return: DataFrame with duplicates removed.
    </details>
    """, unsafe_allow_html=True)
    # Display duplicated values before removal
    duplicated_rows = df[df.duplicated()]
    st.write("Duplicated Rows (Before Removal):")
    st.dataframe(duplicated_rows)
    st.write(f"number of duplicated rows- {duplicated_rows.shape[0]}")

    # Option to remove duplicates or not

    remove_duplicates_option = st.checkbox("Remove duplicates", value=False)
    if remove_duplicates_option:
        # Remove duplicates
        df_cleaned = df.drop_duplicates()
        st.success("Duplicates removed successfully!")
        # Display information about the dataset after removal
        st.write("Number of Rows (After Removal):", df_cleaned.shape[0])
        st.write("Number of Columns (After Removal):", df_cleaned.shape[1])
    else:
        df_cleaned = df
    return df_cleaned


# Function to handle categorical variables
def handle_categorical_variables(df):
    st.markdown(
    """
        <details>
        <summary><strong><div class="info">Handling Categorical Variables:</div></strong></summary>

        <strong>What is Handling Categorical Variables:</strong>
        - Handling categorical variables involves converting them into a format suitable for machine learning algorithms.

        <strong>Where to Use Handling Categorical Variables:</strong>
        - Necessary when machine learning models require numerical input and cannot directly process categorical data.

        <strong>When to Use Handling Categorical Variables:</strong>
        - Before training machine learning models that do not support categorical data.

        <strong>Why We Are Using Handling Categorical Variables:</strong>
        - Enables machine learning models to process and learn from categorical features.
        - Ensures compatibility with algorithms that only accept numerical input.

        <strong>How to Use Handling Categorical Variables:</strong>
        1. Choose the handling method: 'one-hot' encoding or other appropriate techniques.
        2. Call this function with the DataFrame and the chosen method.

        Example:
        ```python
        # One-hot encode categorical variables
        df = handle_categorical_variables(df, method='one-hot')
        # Use other encoding methods if needed
        # df = handle_categorical_variables(df, method='label-encoding')

        ```
        <strong>Note:</strong>
        - 'method' specifies the categorical variable handling technique (default is 'one-hot' encoding).
        - Other techniques like 'label-encoding' can be used based on the nature of the data.
        :param df: The input DataFrame.
        :param method: Categorical variable handling method, choose 'one-hot' encoding or others, default is 'one-hot'.
        :return: DataFrame with handled categorical variables.
        </details>
        """, unsafe_allow_html=True)
    key_value = str(next(key))
    st.subheader("Handling Categorical Variables:")
    # Option to choose method
    method = st.radio("Select method:", ["One-Hot Encoding", "Label Encoding"])
    df_cleaned = df.copy()
    if method == "One-Hot Encoding":
        # One-hot encoding selected columns
        columns_to_encode = st.multiselect("Select columns to one-hot encode:", df.select_dtypes(include="object").columns,key=key_value)
        if columns_to_encode:
            df_cleaned = pd.get_dummies(df_cleaned, columns=columns_to_encode, drop_first=True)
            st.success("One-Hot Encoding applied successfully!")
    elif method == "Label Encoding":
        # Label encoding selected columns
        key_value = str(next(key))
        columns_to_encode = st.multiselect("Select columns to label encode:", df.select_dtypes(include="object").columns,key=key_value)
        if columns_to_encode:
            label_encoder = LabelEncoder()
            df_cleaned[columns_to_encode] = df_cleaned[columns_to_encode].apply(label_encoder.fit_transform)
            st.success("Label Encoding applied successfully!")
    return df_cleaned


# st.dataframe(session_state.selected_dataset)
def modify_columns(df):
    key_value = str(next(key))
    st.subheader("Modify Columns:")
    # Display the current list of columns
    st.write("Current Columns:", df.columns)
    # Option to either drop or select columns
    action = st.radio("Choose action:", ["Drop Columns", "Select Columns"])
    if action == "Drop Columns":
        # Option to select columns to drop
        columns_to_drop = st.multiselect("Select columns to drop:", df.columns,key=key_value)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            st.success("Columns dropped successfully!")
        else:
            st.info("No columns selected to drop.")
    elif action == "Select Columns":
        # Option to select columns to remain
        key_value = str(next(key))
        columns_to_remain = st.multiselect("Select columns to remain:", df.columns,key=key_value)
        if columns_to_remain:
            df = df[columns_to_remain]
            st.success("Columns selected successfully!")
        else:
            st.info("No columns selected to remain.")
    return df


# Function to handle missing values
def handle_missing_values(df):
    st.markdown(
     """
    <details>
    <summary><strong><h3><div class="info">Handle Missing Values:</div></h3></strong></summary>

    <strong>What is Handling Missing Values:</strong>
    - Handling missing values involves addressing cells in a DataFrame with no data.

    <strong>Where to Use Handling Missing Values:</strong>
    - Apply when dealing with datasets containing missing or null values.
    - Essential before analysis or modeling.

    <strong>When to Use Handling Missing Values:</strong>
    - As an initial step in data preprocessing.

    <strong>Why We Are Using Handling Missing Values:</strong>
    - Missing values can distort analysis and modeling results.
    - Appropriate handling ensures robust and accurate analyses.

    <strong>How to Use Handling Missing Values:</strong>
    1. Identify columns or cells with missing values.
    2. Choose a strategy (e.g., 'mean', 'median', 'mode', or 'drop') based on the nature of the missing data.
    3. Call this function with the DataFrame and chosen strategy.

    Example:
    ```python
    # Identify missing values before handling
    print("Before Handling Missing Values:")
    print(df.isnull().sum())
    # Handle missing values
    df = handle_missing_values(df, strategy='mean')
    # Identify missing values after handling
    print("After Handling Missing Values:")
    print(df.isnull().sum())

    ```
    <strong>Note:</strong>
    - Strategies include 'mean', 'median', 'mode', or 'drop'.
    - The function replaces or removes missing values based on the chosen strategy.
    :param df: The input DataFrame.
    :param strategy: The strategy to handle missing values, default is 'mean'.
    :return: DataFrame with missing values handled.
    </details>
    """, unsafe_allow_html=True)
    
    key_value = str(next(key))
    missing_values_summary = df.isnull().sum()
    missing_percentage = (missing_values_summary / len(df)) * 100
    st.write("Missing Values Summary:")
    missing_data_table = pd.DataFrame({
        'Feature': missing_values_summary.index,
        'Missing Values': missing_values_summary.values,
        'Percentage': missing_percentage.values
    })
    # Sort the table by percentage in descending order
    missing_data_table = missing_data_table.sort_values(by='Percentage', ascending=False)
    # Display the table
    st.table(missing_data_table)
    # Option to visualize the distribution of missing values
    if st.checkbox("Visualize Missing Values ", key="visualize_missing_key"):
        # Bar chart showing missing values for each column
        st.subheader("Missing Data Analysis - Bar Chart")
        st.bar_chart(missing_values_summary)
    # Show missing values columns
    st.subheader("Columns with Missing Values:")
    missing_columns = missing_values_summary[missing_values_summary > 0].index
    # Convert Index to list to resolve the issue
    missing_columns_list = missing_columns.tolist()
    selected_columns = st.multiselect("Select Columns", missing_columns_list, default=missing_columns_list,key=key_value)
    # Option to handle missing values column by column
    for col in selected_columns:
        st.subheader(f"Handling Missing Values for {col}")
        # Option to drop or fill missing values
        missing_option = st.radio(f"Select missing values handling option for {col}:", ["Drop Rows", "Drop Column", "Fill"])
        if missing_option == "Drop Rows":
            df = df.dropna(subset=[col])
            st.success(f"Rows with missing values dropped for {col}!")
        elif missing_option == "Drop Column":
            df = df.drop(columns=[col])
            st.success(f"Column {col} dropped due to missing values!")
        else:
            # Option to fill missing values with mean, median, or mode
            key_value = str(next(key))
            fill_option = st.selectbox(f"Select filling method for {col}:", ["Mean", "Median", "Mode"],key=key_value)
            if fill_option == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_option == "Median":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            st.success(f"Missing values filled for {col}!")
    st.write("Number of Rows :", df.shape[0])
    st.write("Number of Columns :", df.shape[1])
    return df


# Function to handle datetime variables
def handle_datetime_variables(df):
    st.markdown(
    """
        <details>
        <summary><strong><h3><div class="info">Handle Datetime Variables:</div></h3></strong></summary>

        <strong>What is Handling Datetime Variables:</strong>
        - Handling datetime variables involves preprocessing date and time columns to make them suitable for analysis.

        <strong>Where to Use Handling Datetime Variables:</strong>
        - When dealing with datasets containing date and time information.
        - Essential for time-series analysis and understanding temporal patterns.

        <strong>When to Use Handling Datetime Variables:</strong>
        - During the data preprocessing phase, especially for time-dependent analyses.

        <strong>Why We Are Using Handling Datetime Variables:</strong>
        - Enables extraction of meaningful insights from temporal data.
        - Facilitates time-series analysis and trend identification.

        <strong>How to Use Handling Datetime Variables:</strong>
        1. Identify columns containing date and time information.
        2. Call this function, providing the DataFrame and a list of datetime columns.

        Example:
        ```python
        # List of datetime columns
        datetime_columns = ['timestamp', 'created_at']
        # Handle datetime variables
        df = handle_datetime_variables(df, datetime_columns)
        ```

        <strong>Note:</strong>
        - This function may involve tasks such as parsing, extracting components, or creating new features from datetime information.
        :param df: The input DataFrame.
        :param datetime_columns: List of column names containing datetime information.
        :return: DataFrame with processed datetime variables.
        </details>
        """, unsafe_allow_html=True)
        
    # Option to choose datetime columns
    key_value = str(next(key))
    datetime_columns = st.multiselect("Select datetime columns:", df.select_dtypes(include="datetime").columns,key=key_value)
    df_cleaned = df.copy()
    if datetime_columns:
        for col in datetime_columns:
            # Extract date and time components
            df_cleaned[col + '_year'] = df_cleaned[col].dt.year
            df_cleaned[col + '_month'] = df_cleaned[col].dt.month
            df_cleaned[col + '_day'] = df_cleaned[col].dt.day
            df_cleaned[col + '_hour'] = df_cleaned[col].dt.hour
            df_cleaned[col + '_minute'] = df_cleaned[col].dt.minute
            df_cleaned[col + '_second'] = df_cleaned[col].dt.second
            st.success(f"Date and time components extracted for {col}!")
    return df_cleaned


def deal_with_skewed_data(df):
    st.markdown(
        """
    <details>
    <summary><strong><h3><div class="info">Dealing with Skewed Data:</div></h3></strong></summary>

    <strong>What is Skewed Data:</strong>
    - Skewed data refers to a distribution where the majority of values are concentrated on one side, causing an imbalance.
    
    <strong>Where to Use Skewed Data Handling:</strong>
    - Applied when numeric features in the dataset exhibit significant skewness, impacting the performance of statistical models.
    
    <strong>When to Use Skewed Data Handling:</strong>
    - Before training regression models, where assumptions of normality are crucial.
    - When dealing with datasets containing features with uneven distributions.
    
    <strong>Why We Are Using Skewed Data Handling:</strong>
    - Aims to normalize the distribution of numeric features, improving model accuracy.
    - Helps models better capture patterns in data with a balanced distribution.
    
    <strong>How to Use Skewed Data Handling:</strong>
    1. Identify numeric columns with skewed distributions.
    2. Call this function, providing the DataFrame and a list of numeric columns.
    
    Example:
    ```python
    # List of numeric columns
    numeric_columns = ['income', 'price']
    # Handle skewed data
    df = handle_skewed_data(df, numeric_columns)
    ```
    
    <strong>Note:</strong>
    - This function may involve techniques such as log transformation to mitigate skewness.
    :param df: The input DataFrame.
    :param numeric_columns: List of column names containing numeric features.
    :return: DataFrame with handled skewed data.
    </details>
    """, unsafe_allow_html=True)

    # Calculate skewness for all numerical columns
    skewness_data_before = pd.DataFrame({
        'Column': df.select_dtypes(include=np.number).columns,
        'Skewness Before': [df[col].skew() for col in df.select_dtypes(include=np.number).columns]
    })
    # Display skewness table before transformation
    st.write("Skewness for All Numerical Columns (Before Transformation):")
    st.table(skewness_data_before)
    # Option to choose columns for skewness transformation
    numerical_columns = df.select_dtypes(include=np.number).columns
    key_value = str(next(key))
    skewed_columns = st.multiselect("Select columns for skewness transformation:", numerical_columns,key=key_value)
    if not skewed_columns:
        st.warning("Please select at least one numerical column for skewness transformation.")
        return df
    for col in skewed_columns:
        st.subheader(f"Dealing with Skewed Data for {col}:")
        # Check if the data is skewed
        skewness_before = df[col].skew()
        if abs(skewness_before) > 1:
            st.write(f"The data in column '{col}' is skewed with skewness: {skewness_before}")
            # Choose skewness transformation method
            transformation_method = st.selectbox(
                f"Select skewness transformation method for {col}:",
                ["Log Transformation", "Square Root Transformation", "Box-Cox Transformation", "Exponential Transformation", "Reciprocal Transformation"]
            )
            # Perform the selected transformation
            if transformation_method == "Log Transformation":
                df[col] = np.log1p(df[col])
                st.success(f"Log transformation applied for {col}!")
            elif transformation_method == "Square Root Transformation":
                df[col] = np.sqrt(df[col])
                st.success(f"Square root transformation applied for {col}!")
            elif transformation_method == "Box-Cox Transformation":
                # Adding a small constant to handle zero or negative values
                df[col], _ = stats.boxcox(df[col] + 1)
                st.success(f"Box-Cox transformation applied for {col}!")
            elif transformation_method == "Exponential Transformation":
                df[col] = np.exp(df[col])
                st.success(f"Exponential transformation applied for {col}!")
            elif transformation_method == "Reciprocal Transformation":
                df[col] = 1 / (df[col] + 1)
                st.success(f"Reciprocal transformation applied for {col}!")
            # Check skewness after transformation
            skewness_after = df[col].skew()
            st.write(f"Skewness after transformation: {skewness_after}")
        else:
            st.write(f"The data in column '{col}' is not significantly skewed (skewness: {skewness_before}). No transformation applied.")
    # Calculate skewness after all transformations
    skewness_data_after = pd.DataFrame({
        'Column': df.select_dtypes(include=np.number).columns,
        'Skewness After': [df[col].skew() for col in df.select_dtypes(include=np.number).columns]
    })
    # Display skewness table after all transformations
    st.write("Skewness for All Numerical Columns (After Transformation):")
    st.table(skewness_data_after)
    return df


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.
    Parameters:
    - df (pd.DataFrame): The DataFrame.
    - columns_to_drop (list): List of columns to drop.
    Returns:
    - pd.DataFrame: The DataFrame after dropping specified columns.
    """
    return df.drop(columns=columns_to_drop, axis=1)

def handle_multicollinearity(df):
    st.markdown(
        """
    <details>
    <summary><strong><h3><div class="info">Handle Multicollinearity:</div></h3></strong></summary>
    
    <strong>What is Multicollinearity:</strong>
    - Multicollinearity occurs when two or more features in a dataset are highly correlated, leading to redundancy in the information they provide.
    
    <strong>Where to Use Multicollinearity Handling:</strong>
    - Applied in regression analysis when predictor variables exhibit high correlation.
    - Essential for maintaining the independence of predictors in statistical models.
    
    <strong>When to Use Multicollinearity Handling:</strong>
    - Before training regression models to avoid unreliable coefficient estimates.
    - When exploring relationships between variables and preventing information redundancy.
    
    <strong>Why We Are Using Multicollinearity Handling:</strong>
    - Improves the stability and reliability of regression models.
    - Prevents inflated standard errors and inaccurate estimation of feature importance.
    
    <strong>How to Use Multicollinearity Handling:</strong>
    1. Set a threshold for correlation (default is 0.8).
    2. Call this function, providing the DataFrame and the correlation threshold.
    
    Example:
    ```python
    # Handle multicollinearity with the default threshold
    df = handle_multicollinearity(df)
    # Handle multicollinearity with a custom threshold
    df = handle_multicollinearity(df, threshold=0.85)
    ```
    
    <strong>Note:</strong>
    - This function identifies highly correlated features and removes one of the correlated pair.
    :param df: The input DataFrame.
    :param threshold: The correlation threshold for identifying multicollinear features (default is 0.8).
    :return: DataFrame with handled multicollinearity.
    </details>
    """, unsafe_allow_html=True)

    def find_multicollinear_columns(correlation_matrix, threshold):
        st.markdown("""
        Find columns with correlation higher than the specified threshold.
        Parameters:
        - correlation_matrix (pd.DataFrame): The correlation matrix.
        - threshold (float): The correlation threshold.
        Returns:
        - list: List of columns with high correlation.
        """, unsafe_allow_html=True)
        # Select upper triangle of correlation matrix
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
        # Find columns with correlation higher than the threshold
        multicollinear_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        return multicollinear_columns

    # Calculate the correlation matrix
    correlation_matrix = df.select_dtypes(include=np.number).corr()
    # Display the correlation matrix
    st.write("Correlation Matrix:")
    st.table(correlation_matrix)
    # Option to set a threshold for identifying multicollinear columns
    correlation_threshold = st.slider("Select correlation threshold:", min_value=0.0, max_value=1.0, value=0.8)
    # Find columns with high correlation
    multicollinear_columns = find_multicollinear_columns(correlation_matrix, correlation_threshold)
    # Display columns with multicollinearity
    st.write("Columns with Multicollinearity:")
    st.write(multicollinear_columns)
    # Option to drop multicollinear columns
    key_value = str(next(key))
    drop_multicollinear_columns = st.checkbox("Drop Multicollinear Columns",key=key_value)
    if drop_multicollinear_columns:
        # Drop multicollinear columns
        df = drop_columns(df, multicollinear_columns)
        st.success("Multicollinear columns dropped!")
    return df


# def handle_imbalanced_data(df, target_column):
#     st.subheader("Handling Imbalanced Data:")
#     # Display class distribution of the target variable
#     class_distribution = df[target_column].value_counts()
#     st.write("Class Distribution:")
#     st.write(class_distribution)
#     # Option to perform under-sampling or over-sampling
#     sampling_method = st.radio("Select Sampling Method:", ["Under-sampling (NearMiss)", "Over-sampling (RandomOverSampler)"])
#     if sampling_method == "Under-sampling (NearMiss)":
#         # Perform under-sampling using NearMiss
#         nearmiss_version = st.selectbox("Select NearMiss Version:", ["1", "2", "3"])
#         nearmiss = NearMiss(version=int(nearmiss_version))
#         X_resampled, y_resampled = nearmiss.fit_resample(df.drop(columns=[target_column]), df[target_column])
#         df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
#         st.success("Under-sampling completed!")
#     elif sampling_method == "Over-sampling (RandomOverSampler)":
#         # Perform over-sampling using RandomOverSampler
#         oversample_ratio = st.slider("Select Over-sampling Ratio:", min_value=1.0, max_value=5.0, value=1.0)
#         random_oversampler = RandomOverSampler(sampling_strategy=oversample_ratio)
#         X_resampled, y_resampled = random_oversampler.fit_resample(df.drop(columns=[target_column]), df[target_column])
#         df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
#         st.success("Over-sampling completed!")
#     # Display class distribution after sampling
#     class_distribution_resampled = df_resampled[target_column].value_counts()
#     st.write("Class Distribution After Sampling:")
#     st.write(class_distribution_resampled)
#     return df_resampled

# # Call the function
# session_state.selected_dataset = handle_imbalanced_data(session_state.selected_dataset, target_column="your_target_column_name")
# # Display the dataset after handling imbalanced data
# st.subheader("Dataset After Handling Imbalanced Data:")
# st.dataframe(session_state.selected_dataset)


# Function for feature engineering
def feature_engineering(df):
    st.subheader("Feature Engineering:")
    # Option to choose columns for feature engineering
    key_value = str(next(key))
    feature_columns = st.multiselect("Select columns for feature engineering:", df.columns,key=key_value)
    df_engineered = df.copy()
    if feature_columns:
        for col in feature_columns:
            # Example: Creating a new feature based on existing features (e.g., combining two columns)
            if "feature_combination" in feature_columns:
                df_engineered["feature_combination"] = df_engineered["column1"] * df_engineered["column2"]
                st.success("New feature 'feature_combination' created!")
            # Example: Transforming a numerical feature using a mathematical function
            if "log_transformed_feature" in feature_columns:
                df_engineered["log_transformed_feature"] = np.log1p(df_engineered["numerical_feature"])
                st.success("New feature 'log_transformed_feature' created!")
            # Example: Creating dummy variables for a categorical feature
            if "categorical_feature_dummy" in feature_columns:
                df_engineered = pd.get_dummies(df_engineered, columns=["categorical_feature"], drop_first=True)
                st.success("Dummy variables created for 'categorical_feature'!")
    return df_engineered



# Function to handle outliers
def handle_outliers(df):
    st.markdown("""
    <details>
    <summary><strong><h3><div class="info">handle outliers:</div></h3></strong></summary>
    
    <strong>What is Dealing with Outliers:</strong>
    - Dealing with outliers involves addressing extreme values in a dataset.
    
    <strong>Where to Use Dealing with Outliers:</strong>
    - Apply when datasets contain values significantly deviating from the norm.
    - Crucial for robust analysis and modeling.
    
    <strong>When to Use Dealing with Outliers:</strong>
    - Before analysis or modeling to prevent outliers from skewing results.
    
    <strong>Why We Are Using Dealing with Outliers:</strong>
    - Outliers can significantly impact statistical analyses and machine learning models.
    - Proper handling ensures more accurate and reliable results.
    
    <strong>How to Use Dealing with Outliers:</strong>
    1. Identify columns or variables suspected of containing outliers.
    2. Choose a method (e.g., 'z-score' or 'IQR') for detecting and handling outliers.
    3. Set a threshold to determine what is considered an outlier.
    4. Call this function with the DataFrame, chosen method, and threshold.
    
    Example:
    ```python
    # Identify outliers before handling
    print("Before Dealing with Outliers:")
    print(df.describe())
    # Deal with outliers using z-score method
    df = deal_with_outliers(df, method='z-score', threshold=3)
    # Identify outliers after handling
    print("After Dealing with Outliers:")
    print(df.describe())
    ```
    
    <strong>Note:</strong>
    - Methods include 'z-score' or 'IQR' (Interquartile Range).
    - The function either removes or transforms outliers based on the chosen method and threshold.
    :param df: The input DataFrame.
    :param method: The method for detecting and handling outliers, default is 'z-score'.
    :param threshold: The threshold to determine outliers, default is 3.
    :return: DataFrame with outliers dealt with.
    </details>
    """, unsafe_allow_html=True)

    # Display outliers using different methods
    outlier_method = st.radio("Select outlier detection method:", ["Z-score", "IQR", "Boxplot"])
    st.subheader(f"Outliers Detection using {outlier_method}")
    if outlier_method == "Z-score":
        # Outlier detection using Z-score
        z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
        outliers = (np.abs(z_scores) > 3)
        st.subheader("Outlier Detection - Z-score")
        st.write("Outliers:", outliers.sum())
    elif outlier_method == "IQR":
        # Outlier detection using IQR
        Q1 = df[df.select_dtypes(include=['float64', 'int64'])].quantile(0.25)
        Q3 = df[df.select_dtypes(include=['float64', 'int64'])].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        st.subheader("Outlier Detection - IQR")
        st.write("Outliers:", outliers.sum())
        
    elif outlier_method == "Boxplot":
       # Outlier detection using Box Plot
        st.subheader("Outlier Detection - Box Plot")
        # Allow users to select columns for box plot
        key_value = str(next(key))
        selected_columns_box = st.multiselect("Select Columns for Box Plot", df.columns,key=key_value)
        # Documentation
        st.info(
            "Customize the appearance of the box plots using the provided options. "
            "After selecting columns, the box plots will be displayed below."
        )
        # Display box plots for selected columns one by one
        for column in selected_columns_box:
            st.write(f"**Box Plot for {column}:**")
            try:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[column])
                st.pyplot(fig)
                # Export Options
                st.markdown(get_save_button(fig, f"{column}_boxplot.png"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred while plotting the box plot for {column}: {e}")
        # Error Handling
        if not selected_columns_box:
            st.warning("Please select at least one column for the box plot.")
    # Option to drop outliers
    drop_outliers_option = st.checkbox("Drop Outliers", value=False)
    if drop_outliers_option:
        df = df[~outliers.any(axis=1)]
        st.success("Outliers dropped successfully!")
        st.write("Number of Rows (After droping):", df.shape[0])
        st.write("Number of Columns (After droping):", df.shape[1])
    return df



# Function to standardize or normalize data
def standardize_normalize_data(df):
    st.markdown(
        """
    <details>
    <summary><strong><h3><div class="info">Standardizing/Normalizing Data:</div></h3></strong></summary>

    <strong>What is Standardizing/Normalizing Data:</strong>
    - Standardizing or normalizing involves transforming numerical data to a common scale.
    
    <strong>Where to Use Standardizing/Normalizing Data:</strong>
    - Apply when features have different scales, ensuring fair comparisons in machine learning.
    - Essential for algorithms sensitive to scale, like k-Nearest Neighbors or Support Vector Machines.
    
    <strong>When to Use Standardizing/Normalizing Data:</strong>
    - Before training machine learning models to prevent features with larger scales dominating the learning process.
    
    <strong>Why We Are Using Standardizing/Normalizing Data:</strong>
    - Ensures fair comparison and prevents certain features from having undue influence.
    - Facilitates convergence for gradient-based optimization algorithms.
    
    <strong>How to Use Standardizing/Normalizing Data:</strong>
    1. Specify the columns to standardize/normalize (default is all numerical columns).
    2. Choose the method: 'z-score' (standardization) or 'min-max' (normalization).
    3. Call this function with the DataFrame, columns, and method.
    
    Example:
    ```python
    # Standardize data using z-score
    df = standardize_normalize_data(df, columns=['feature1', 'feature2'], method='z-score')
    # Normalize data using min-max scaling
    df = standardize_normalize_data(df, columns=['feature3', 'feature4'], method='min-max')
    ```
    
    <strong>Note:</strong>
    - 'columns' specifies which numerical columns to standardize/normalize (default is None, meaning all numerical columns).
    - 'method' determines the transformation method ('z-score' for standardization or 'min-max' for normalization).
    :param df: The input DataFrame.
    :param columns: Columns to standardize/normalize, default is None (all numerical columns).
    :param method: Transformation method, choose 'z-score' for standardization or 'min-max' for normalization, default is 'z-score'.
    :return: DataFrame with standardized/normalized data.
    </details>
    """, unsafe_allow_html=True)
    
    # Option to choose method
    method = st.radio("Select method:", ["Standardization", "Normalization"])
    if method == "Standardization":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    # Standardize or normalize selected columns
    key_value = str(next(key))
    columns_to_normalize = st.multiselect("Select columns to standardize/normalize:", df.columns,key=key_value)
    df_cleaned = df.copy()
    if columns_to_normalize:
        df_cleaned[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    st.success("Data standardized/normalized successfully!")
    return df_cleaned


# Function to create an interactive scatter plot
def scatter_plot(df, x_column, y_column, color_column=None):
    st.markdown( """
    <details>
    <summary><strong><h3><div class="info">Scatter Plot:</div></h3></strong></summary>
    
    <strong>What is a Scatter Plot:</strong>
    - A scatter plot is a two-dimensional data visualization that displays individual data points on a graph.
    
    <strong>Where to Use Scatter Plot:</strong>
    - Ideal for visualizing the relationship between two continuous variables.
    - Useful for identifying patterns, trends, and outliers in the data.
    
    <strong>When to Use Scatter Plot:</strong>
    - During exploratory data analysis (EDA) to understand the correlation between two variables.
    - When investigating the presence of clusters or groups in the data.
    
    <strong>Why Use Scatter Plot:</strong>
    - Reveals the distribution and relationship between two variables.
    - Highlights potential correlations and patterns in the data.
    
    <strong>How to Use Scatter Plot:</strong>
    1. Select two continuous variables for the x-axis and y-axis.
    2. Call this function with the dataset and chosen columns.
    
    Example:
    ```python
    # Create a scatter plot
    scatter_plot(data=df, x_col='feature1', y_col='feature2')
    ```
    <strong>Note:</strong>
    - Adjust plot aesthetics and markers for better interpretability.
    - Interpret the spread and concentration of points to derive insights.
    :param data: The DataFrame containing the data.
    :param x_col: The column representing the x-axis variable.
    :param y_col: The column representing the y-axis variable.
    </details>
    """,unsafe_allow_html=True)

    # Allow user to select columns
    key_value = str(next(key))
    x_column = st.selectbox("Select X-axis column:", df.columns, index=df.columns.get_loc(x_column) if x_column in df.columns else 0, key=key_value)
    key_value = str(next(key))
    y_column = st.selectbox("Select Y-axis column:", df.columns, index=df.columns.get_loc(y_column) if y_column in df.columns else 1, key=key_value)
    key_value = str(next(key))
    color_column = st.selectbox("Select Color column (optional):", [None] + list(df.columns), index=df.columns.get_loc(color_column) + 1 if color_column in df.columns else 0, key=key_value)
    # Create the interactive scatter plot using Plotly Express
    fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=f"Scatter Plot: {x_column} vs {y_column}")
    # Show the interactive plot
    st.plotly_chart(fig)
    # Function to create an interactive line chart

def line_chart(df, x_column, y_column, hue=None):
    st.markdown( """
    <details>
    <summary><strong><h3><div class="info">Line Chart:</div></h3></strong></summary>
    
    <strong>What is a Line Chart:</strong>
    - A line chart is a data visualization that displays data points connected by straight line segments.
    
    <strong>Where to Use Line Chart:</strong>
    - Suitable for showing trends and changes over a continuous interval.
    - Commonly used in time series data to illustrate temporal patterns.
    
    <strong>When to Use Line Chart:</strong>
    - When visualizing the progression of a variable over time.
    - Comparing trends between multiple series.
    
    <strong>Why Use Line Chart:</strong>
    - Highlights trends, patterns, or fluctuations in data.
    - Effective for conveying changes and developments.
    
    <strong>How to Use Line Chart:</strong>
    1. Select a continuous variable for the x-axis (usually time).
    2. Choose one or more variables for the y-axis.
    3. Call this function with the dataset and chosen columns.
    
    Example:
    ```python
    # Create a line chart
    line_chart(data=df, x_col='date', y_col='sales', title='Monthly Sales Trends')
    ```
    
    <strong>Note:</strong>
    - Customize labels, colors, and other aesthetics for better clarity.
    - Interpret the slope and direction of lines to derive insights.
    :param data: The DataFrame containing the data.
    :param x_col: The column representing the x-axis variable.
    :param y_col: The column(s) representing the y-axis variable(s).
    :param title: (Optional) Title for the chart.
    :return: Display the line chart.
    </details>
    """,unsafe_allow_html=True)

    # Allow user to select columns
    key_value = str(next(key))
    x_column = st.selectbox("Select X-axis column:", df.columns, index=df.columns.get_loc(x_column) if x_column in df.columns else 0, key=key_value)
    key_value = str(next(key))
    y_column = st.selectbox("Select Y-axis column:", df.columns, index=df.columns.get_loc(y_column) if y_column in df.columns else 1, key=key_value)
    key_value = str(next(key))
    hue = st.selectbox("Select Hue column (optional):", [None] + list(df.columns), index=df.columns.get_loc(hue) + 1 if hue in df.columns else 0, key=key_value)
    # Create the interactive line chart using Plotly Express
    fig = px.line(df, x=x_column, y=y_column, color=hue, title=f"Line Chart: {x_column} vs {y_column}")
    # Show the interactive plot
    st.plotly_chart(fig)

# Function to create an interactive bar chart
def bar_chart(df, x_column, y_column, hue=None):
    st.markdown( """
    <details>
    <summary><strong><h3><div class="info">Bar chart:</div></h3></strong></summary>
    
    <strong>What is a Bar Plot:</strong>
    - A bar plot is a visual representation of data using rectangular bars or columns.
    
    <strong>Where to Use Bar Plot:</strong>
    - Suitable for comparing quantities of different categories.
    - Effective for displaying categorical data.
    
    <strong>When to Use Bar Plot:</strong>
    - When comparing values across distinct categories.
    - Displaying frequencies, counts, or percentages.
    
    <strong>Why Use Bar Plot:</strong>
    - Provides a clear visual comparison between categories.
    - Easy interpretation of relative sizes or frequencies.
    
    <strong>How to Use Bar Plot:</strong>
    1. Select a categorical variable for the x-axis.
    2. Choose a numerical variable for the y-axis.
    3. Call this function with the dataset and chosen columns.
    
    Example:
    ```python
    # Create a bar plot
    bar_plot(data=df, x_col='category', y_col='sales', title='Category-wise Sales Comparison')
    ```
    
    <strong>Note:</strong>
    - Customize labels, colors, and other aesthetics for better clarity.
    - Consider horizontal bar plots for better readability with long category names.
    :param data: The DataFrame containing the data.
    :param x_col: The column representing the x-axis categorical variable.
    :param y_col: The column representing the y-axis numerical variable.
    :param title: (Optional) Title for the chart.
    :return: Display the bar plot.
    </details>
    """,unsafe_allow_html=True)

    # Allow user to select columns
    key_value = str(next(key))
    x_column = st.selectbox("Select X-axis column:", df.columns, index=df.columns.get_loc(x_column) if x_column in df.columns else 0, key=key_value)
    key_value = str(next(key))
    y_column = st.selectbox("Select Y-axis column:", df.columns, index=df.columns.get_loc(y_column) if y_column in df.columns else 1, key=key_value)
    key_value = str(next(key))
    hue = st.selectbox("Select Hue column (optional):", [None] + list(df.columns), index=df.columns.get_loc(hue) + 1 if hue in df.columns else 0, key=key_value)
    # Create the interactive bar chart using Plotly Express
    fig = px.bar(df, x=x_column, y=y_column, color=hue, title=f"Bar Chart: {x_column} vs {y_column}")
    # Show the interactive plot
    st.plotly_chart(fig)

# Function to create an interactive histogram with KDE
def histogram(df, column_name, color_hist="blue", bins_hist=10, hue_column_hist=None):
    st.markdown( """
    <details>
    <summary><strong><h3><div class="info">Histogram:</div></h3></strong></summary>
    
    <strong>What is a Histogram:</strong>
    - A histogram is a graphical representation of the distribution of a dataset.
    
    <strong>Where to Use Histogram:</strong>
    - Used to show the underlying frequency distribution of a continuous variable.
    
    <strong>When to Use Histogram:</strong>
    - To visualize the central tendency, spread, and skewness of a variable.
    - Identify the presence of outliers or unusual patterns.
    
    <strong>Why Use Histogram:</strong>
    - Provides insights into the shape of the data distribution.
    - Highlights concentration and variability in the dataset.
    
    <strong>How to Use Histogram:</strong>
    1. Select a numerical variable to create the histogram.
    2. Choose the number of bins for grouping the data.
    3. Call this function with the dataset and the selected column.
    
    Example:
    ```python
    # Create a histogram
    histogram(data=df, column='age', bins=20, title='Age Distribution')
    ```
    
    <strong>Note:</strong>
    - Adjust the number of bins for better granularity.
    - Interpret the shape of the histogram (normal, skewed, bimodal, etc.).
    :param data: The DataFrame containing the data.
    :param column: The column representing the variable for the histogram.
    :param bins: (Optional) Number of bins for grouping the data.
    :param title: (Optional) Title for the chart.
    :return: Display the histogram.
    </details>
    """,unsafe_allow_html=True)

    # Allow user to select the column, number of bins, and KDE option
    column_name = st.selectbox("Select column:", df.columns, key="histogram_column")
    color_hist = st.selectbox("Select Color for Histogram", ["blue", "green", "red", "purple", "orange"], key="histogram_color")
    bins_hist = st.slider("Select Number of Bins", min_value=5, max_value=50, value=10, key="histogram_bins")
    hue_column_hist = st.selectbox("Select Hue Column for Histogram", [None] + df.columns.tolist(), index=0, key="histogram_hue")
    fig, ax = plt.subplots()
    sns.histplot(
        data=df,  # Specify the DataFrame explicitly
        x=column_name,
        kde=True,
        bins=bins_hist,
        color=color_hist,
        hue=hue_column_hist,
    )
    # Add labels and title
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram: {column_name}")
    st.pyplot(fig)
 
# Function to create an interactive box plot
def box_plot(df, x_column, y_column, hue=None):
    st.markdown(  """
    <details>
    <summary><strong><h3><div class="info">Boxplot:</div></h3></strong></summary>
    
    <strong>What is a Boxplot?</strong>
    - A boxplot displays the distribution of a dataset and its central tendency.
    
    <strong>Where to Use Boxplot?</strong>
    - Use boxplots to identify the spread, skewness, and potential outliers in a numerical variable.
    
    <strong>When to Use Boxplot?</strong>
    - Boxplots are suitable for comparing distributions or visualizing the spread of data.
    
    <strong>Why Use Boxplot?</strong>
    - Boxplots provide a concise summary of the distribution, including median, quartiles, and outliers.
    
    <strong>How to Use Boxplot?</strong>
    1. Provide the dataset (`data`) and the columns for the x-axis (`x_col`) and y-axis (`y_col`).
    2. Call this function, passing the required arguments.
    3. Customize the boxplot as needed for better visualization.
    
    Example:
    ```python
    # Example dataset and column names
    boxplot_data = your_dataset_here
    x_column = 'categorical_column'
    y_column = 'numerical_column'
    # Create and display the boxplot
    create_boxplot(boxplot_data, x_column, y_column)
    ```
    
    <strong>Note:</strong>
    - Ensure the dataset and column names match your actual data structure.
    </details>
    """,unsafe_allow_html=True)

    # Allow user to select columns
    key_value = str(next(key))
    x_column = st.selectbox("Select X-axis column:", df.columns, index=df.columns.get_loc(x_column) if x_column in df.columns else 0, key=key_value)
    key_value = str(next(key))
    y_column = st.selectbox("Select Y-axis column:", df.columns, index=df.columns.get_loc(y_column) if y_column in df.columns else 1, key=key_value)
    key_value = str(next(key))
    hue = st.selectbox("Select Hue column (optional):", [None] + list(df.columns), index=df.columns.get_loc(hue) + 1 if hue in df.columns else 0, key=key_value)
    # Create the interactive box plot using Plotly Express
    fig = px.box(df, x=x_column, y=y_column, color=hue, title=f"Box Plot: {x_column} vs {y_column}")
    # Show the interactive plot
    st.plotly_chart(fig)

# Function to create a heatmap
def heatmap(df):
    st.markdown( """
    <details>
    <summary><strong><h3><div class="info">Heatmap:</div></h3></strong></summary>
    
    <strong>What is a Heatmap?</strong>
    - A heatmap is a graphical representation of data where values in a matrix are represented as colors.
    
    <strong>Where to Use Heatmap?</strong>
    - Heatmaps are used to visualize the magnitude of a phenomenon as color in two dimensions.
    
    <strong>When to Use Heatmap?</strong>
    - Use heatmaps to display relationships between two categorical variables or display correlations.
    
    <strong>Why Use Heatmap?</strong>
    - Heatmaps provide an intuitive way to identify patterns and trends within complex datasets.
    
    <strong>How to Use Heatmap?</strong>
    1. Prepare a dataset with relevant values.
    2. Call this function, providing the dataset as an argument.
    3. Customize the heatmap for clarity (adjust color maps, labels, etc.).
    
    Example:
    ```python
    # Example dataset
    heatmap_data = your_dataset_here
    # Create and display the heatmap
    create_heatmap(heatmap_data)
    ```
    
    <strong>Note:</strong>
    - Ensure the dataset is well-organized for meaningful interpretation.
    </details>
    """,unsafe_allow_html=True)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    # Add title
    plt.title("Correlation Matrix Heatmap")
    st.pyplot()

# Function to create pair plots
def pair_plot(df, hue=None):
    st.markdown( """
    <details>
    <summary><strong><h3><div class="info">Pairplot:</div></h3></strong></summary>
    
    <strong>What is a Pairplot:</strong>
    - A pairplot is a matrix of scatterplots showing relationships between paired variables.
    
    <strong>Where to Use Pairplot:</strong>
    - Used to visualize pairwise relationships in a dataset.
    
    <strong>When to Use Pairplot:</strong>
    - To explore correlations and patterns between multiple variables.
    
    <strong>Why Use Pairplot:</strong>
    - Identifies potential patterns, clusters, or outliers in multivariate data.
    
    <strong>How to Use Pairplot:</strong>
    1. Provide the dataset with the selected variables.
    2. Specify the target variable for hue (optional).
    3. Call this function to generate the pairplot.
    
    Example:
    ```python
    # Create a pairplot
    pairplot(data=df, hue='category', title='Pairplot')
    ```
    
    <strong>Note:</strong>
    - Interpret scatterplots for relationships between variables.
    - Use hue for visualizing relationships with a categorical variable.
    :param data: The DataFrame containing the data.
    :param hue: (Optional) Categorical variable for color differentiation.
    :param title: (Optional) Title for the chart.
    :return: Display the pairplot.
    </details>
    """,unsafe_allow_html=True)

    # Allow user to select columns
    key_value = str(next(key))
    hue = st.selectbox("Select Hue column (optional):", [None] + list(df.columns), index=df.columns.get_loc(hue) + 1 if hue in df.columns else 0,key=key_value)
    # Plot the pair plot
    sns.pairplot(df, hue=hue)
    st.pyplot()

# Function to create an interactive pie chart
def pie_chart(df, column_name):
    st.markdown( 
        """
    <details>
    <summary><strong><h3><div class="info">Pie Chart:</div></h3></strong></summary>


    <strong>What is a Pie Chart:</strong>
    - A pie chart is a circular statistical graphic that is divided into slices to illustrate numerical proportions.

    <strong>Where to Use Pie Chart:</strong>
    - Suitable for displaying the distribution of a categorical variable.

    <strong>When to Use Pie Chart:</strong>
    - When you want to show the composition of a whole in terms of percentages.

    <strong>Why Use Pie Chart:</strong>
    - Easily conveys the proportion of each category in a dataset.

    <strong>How to Use Pie Chart:</strong>
    1. Provide the dataset and the categorical column.
    2. Call this function to generate the pie chart.

    Example:
    ```python
    # Create a pie chart
    pie_chart(data=df, column='category', title='Pie Chart')

    ```
    <strong>Note:</strong>
    - Use for visualizing the distribution of a single categorical variable.
    - Limit the number of categories for clarity.

    :param data: The DataFrame containing the data.
    :param column: The categorical column for the pie chart.
    :param title: (Optional) Title for the chart.
    :return: Display the pie chart.
    </details>

    """,unsafe_allow_html=True)
    # Allow user to select columns
    key_value = str(next(key))
    column_name = st.selectbox("Select Column for Pie Chart:", df.columns, index=df.columns.get_loc(column_name) if column_name in df.columns else 0,key=key_value)
    # Create the interactive pie chart using Plotly Express with custom color intensity
    fig = px.pie(
        df,
        names=column_name,
        hole=st.slider("Hole Size", 0.0, 0.5, 0.2, step=0.05, key="hole_size"),
        labels={"names": "Categories", "value": "Count"},
        color_discrete_sequence=px.colors.qualitative.Set3,
        color_discrete_map={column_name: "rgba(144, 238, 144, 1.0)"},  # Adjust the color here
    )
    # Show the pie chart
    st.plotly_chart(fig)


# Function to create a violin plot
def violin_plot(df):
    st.markdown("""
    <details>
    <summary><strong><h3><div class="info">Violin Plot:</div></h3></strong></summary>

    <strong>What is a Violin Plot:</strong>
    - A violin plot is used to visualize the distribution and probability density of multiple variables.
    
    <strong>Where to Use Violin Plot:</strong>
    - Effective for comparing the distribution of different categories.
    
    <strong>When to Use Violin Plot:</strong>
    - When you want to understand the distribution of a numerical variable across different groups or categories.
    
    <strong>Why Use Violin Plot:</strong>
    - Combines aspects of a box plot and a kernel density plot, providing more information about the distribution.
    
    <strong>How to Use Violin Plot:</strong>
    1. Provide the dataset and specify the columns for x-axis and y-axis.
    2. Call this function to generate the violin plot.
    
    Example:
    ```python
    # Create a violin plot
    violin_plot(data=df, x_col='category', y_col='value', title='Violin Plot')
    ```
    
    <strong>Note:</strong>
    - Useful for visualizing the distribution of a numerical variable within different groups.
    - Helps identify variations in data distribution.
    :param data: The DataFrame containing the data.
    :param x_col: The categorical column for the x-axis.
    :param y_col: The numerical column for the y-axis.
    :param title: (Optional) Title for the chart.
    :return: Display the violin plot.
    </details>
    """ ,unsafe_allow_html=True)

    # Allow user to select columns
    x_column_options = list(df.columns)
    x_column = st.selectbox("Select X-axis column:", x_column_options, index=0)
    y_column_options = [None] +list(df.columns)
    y_column = st.selectbox("Select Y-axis column:", y_column_options, index=1)
    hue_options = [None] + list(df.columns)
    hue = st.selectbox("Select Hue column (optional):", hue_options, index=0)
    # Plot the violin plot using Plotly Express
    fig = px.violin(
        df, 
        x=x_column, 
        y=y_column, 
        color=hue,
        box=True,  # Include box plot inside the violin
        points="all",  # Show all data points
        hover_data=df.columns,
        title=f"Violin Plot: {x_column} vs {y_column}",
    )
    # Customize layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=hue,
        autosize=False,
        width=800,
        height=500,
    )
    # Show the plot
    st.plotly_chart(fig)

# Function to create a 3D scatter plot
def scatter_3d(df, x_column, y_column, z_column, color_column=None, size_column=None):
    st.subheader("3D Scatter Plot")
    # Allow user to select columns
    key_value = str(next(key))
    x_column = st.selectbox("Select X-axis column:", df.columns, index=df.columns.get_loc(x_column) if x_column in df.columns else 0,key=key_value)
    key_value = str(next(key))
    y_column = st.selectbox("Select Y-axis column:", df.columns, index=df.columns.get_loc(y_column) if y_column in df.columns else 1,key=key_value)
    key_value = str(next(key))
    z_column = st.selectbox("Select Z-axis column:", df.columns, index=df.columns.get_loc(z_column) if z_column in df.columns else 2,key=key_value)
    key_value = str(next(key))
    color_column = st.selectbox("Select Color column (optional):", [None] + list(df.columns), index=df.columns.get_loc(color_column) + 1 if color_column in df.columns else 0,key=key_value)
    key_value = str(next(key))
    size_column = st.selectbox("Select Size column (optional):", [None] + list(df.columns), index=df.columns.get_loc(size_column) + 1 if size_column in df.columns else 0,key=key_value)
    # Plot the 3D scatter plot
    fig = px.scatter_3d(df, x=x_column, y=y_column, z=z_column, color=color_column, size=size_column)
    # Add title
    fig.update_layout(title_text=f"3D Scatter Plot: {x_column} vs {y_column} vs {z_column}")
    st.plotly_chart(fig)

# Function to create a time series plot
def time_series_plot(df, date_column, value_column):
    st.subheader("Time Series Plot")
    # Allow user to select columns
    key_value = str(next(key))
    date_column = st.selectbox("Select Date column:", df.columns, index=df.columns.get_loc(date_column) if date_column in df.columns else 0,key=key_value)
    key_value = str(next(key))
    value_column = st.selectbox("Select Value column:", df.columns, index=df.columns.get_loc(value_column) if value_column in df.columns else 1,key=key_value)
    # Plot the time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_column], df[value_column])
    # Add labels and title
    plt.xlabel(date_column)
    plt.ylabel(value_column)
    plt.title(f"Time Series Plot: {value_column} over time")
    st.pyplot()

# Function to create a geographic map
def geo_map(df, lat_column, lon_column, size_column=None, color_column=None, hover_data=None):
    st.subheader("Geographic Map")
    # Allow user to select columns
    key_value = str(next(key))
    lat_column = st.selectbox("Select Latitude column:", df.columns, index=df.columns.get_loc(lat_column) if lat_column in df.columns else 0,key=key_value)
    key_value = str(next(key))
    lon_column = st.selectbox("Select Longitude column:", df.columns, index=df.columns.get_loc(lon_column) if lon_column in df.columns else 1,key=key_value)
    key_value = str(next(key))
    size_column = st.selectbox("Select Size column (optional):", [None] + list(df.columns), index=df.columns.get_loc(size_column) + 1 if size_column in df.columns else 0,key=key_value)
    key_value = str(next(key))
    color_column = st.selectbox("Select Color column (optional):", [None] + list(df.columns), index=df.columns.get_loc(color_column) + 1 if color_column in df.columns else 0,key=key_value)
    key_value = str(next(key))
    hover_data = st.multiselect("Select additional data for hover (optional):", list(df.columns), default=hover_data,key=key_value)
    # Plot the geographic map
    fig = px.scatter_geo(df, lat=lat_column, lon=lon_column, size=size_column, color=color_column, hover_data=hover_data)
    # Add title
    fig.update_layout(title_text="Geographic Map")
    st.plotly_chart(fig)

# Function to drop selected rows
def drop_selected_rows(df):
    st.subheader("Drop Selected Rows")

    # Display the current dataset for reference
    st.write("Current Dataset:")
    st.dataframe(df)

    # Allow the user to select rows to drop
    rows_to_drop = st.multiselect("Select rows to drop:", df.index.tolist())

    # Drop the selected rows
    if rows_to_drop:
        df = df.drop(rows_to_drop)
        st.success("Selected rows dropped successfully!")
    else:
        st.warning("No rows selected to drop.")

    # Display the updated dataset
    st.write("Updated Dataset:")
    st.dataframe(df)

    return df





# Sidebar for navigation
st.sidebar.title("Navigation")

st.sidebar.subheader(" Data Loading")

# Load a sample dataset for initial display
st.sidebar.subheader("Load the Dataset")
# Using Seaborn's built-in 'tips' dataset for demonstration
default_dataset = sns.load_dataset("tips")

# Option to upload custom CSV file
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
# Option to choose from preloaded datasets
preloaded_datasets = {
    "Tips Dataset": sns.load_dataset("tips"),
    "Iris Dataset": sns.load_dataset("iris"),
    "Titanic Dataset": sns.load_dataset("titanic"),
}
selected_dataset = st.sidebar.selectbox("Choose a preloaded dataset", list(preloaded_datasets.keys()))
# Load the selected dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded successfully!")
    session_state.selected_dataset = df  # Save in session state
elif selected_dataset:
    df = preloaded_datasets[selected_dataset]
    st.sidebar.success(f"Preloaded dataset '{selected_dataset}' loaded successfully!")
    session_state.selected_dataset = df  # Save in session state
# Display the selected dataset in the sidebar
st.sidebar.subheader("Selected Dataset:")
st.sidebar.dataframe(session_state.selected_dataset)
# Descriptive Statistics and Data Profiling
if session_state.selected_dataset is not None:
    # Data Exploration Section
    st.title("Data Exploration")

    # Summary Statistics Section
    st.subheader("1. Data Information:")
    st.write("Number of Rows:", df.shape[0])
    st.write("Number of Columns:", df.shape[1])

    buffer = io.StringIO()
    df.info(buf=buffer)
    # Get the buffer content
    info_output = buffer.getvalue()
    st.write("**Dataset info:**")
    st.text(info_output)
    st.write("**Descriptive Statistics:**")
    st.write(session_state.selected_dataset.describe(include="all"))

    st.write("**Data Profiling:**")
    # Display data types, unique values, and null values for each column
    data_profile = pd.DataFrame({
        'Data Type': session_state.selected_dataset.dtypes,
        'Unique Values': session_state.selected_dataset.nunique(),
        'Null Values': session_state.selected_dataset.isnull().sum()
    })
    st.dataframe(data_profile)
st.write("---")

def show_value_counts(df):
    st.markdown("#### Value counts")
    selected_columns = st.multiselect("Select Columns for Value Counts", df.columns,help="Select the columns")
    
    if not selected_columns:
        st.warning("Please select at least one column for value counts.")
        return

    all_categorical = all(pd.api.types.is_categorical_dtype(df[col]) for col in selected_columns)

    if all_categorical:
        value_counts_result = df.groupby(selected_columns).size().reset_index(name='Count')
    else:
        value_counts_result = df[selected_columns].value_counts().reset_index(name='Count')

    st.write(f"Value Counts for {selected_columns}:")
    st.write(value_counts_result)

# Call the function
show_value_counts(session_state.selected_dataset)


# Create two columns
col1, col2 = st.columns(2)

# Function to perform group by
def perform_group_by(df):
    col1.markdown("#### Group by")
    selected_column_gb = col1.selectbox("Select Column for Group By", df.columns)
    aggregation_function_gb = col1.selectbox("Select Aggregation Function", ["mean", "sum", "count", "min", "max"])
    grouped_data = df.groupby(selected_column_gb).agg(aggregation_function_gb).reset_index()
    col1.write(f"Grouped Data by {selected_column_gb} using {aggregation_function_gb}:")
    col1.write(grouped_data)

# Call the function
perform_group_by(session_state.selected_dataset)

# Function to show unique values
def show_unique_values(df):
    col2.markdown("#### Unique values")
    selected_column_uv = col2.selectbox("Select Column for Unique Values", df.columns)
    unique_values_result = df[selected_column_uv].unique()
    col2.write(f"Unique Values for {selected_column_uv}:")
    col2.write(unique_values_result)

# Call the function
show_unique_values(session_state.selected_dataset)



st.write("---")


st.subheader("Data Cleaning Operations and Visualizations")
st.sidebar.subheader("data cleaning")
selected_operations = st.sidebar.multiselect(
    "Select data cleaning operations:",
    ["Drop Selected Rows","Convert Data Types", "Remove Duplicates", "Handle Categorical Variables", "Modify Columns",
     "Handle Missing Values", "Handle Datetime Variables", "Deal with Skewed Data", "Handle Multicollinearity",
     "Handle Imbalanced Data", "Feature Engineering", "Handle Outliers", "Standardize/Normalize Data"]
)
# Step 5: Visualization Operations
st.sidebar.subheader("Visualization Operations")
selected_visualizations = st.sidebar.multiselect(
    "Select visualizations:",
    ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap", "Pair Plot", 
     "Pie Chart", "Violin Plot", "3D Scatter Plot", "Time Series Plot", "Geographic Map"]
)
st.sidebar.write("---")
# Data Cleaning Operations Based on Selection
for selected_operation in selected_operations:
    if selected_operation == "Convert Data Types":
        session_state.selected_dataset = convert_data_types(session_state.selected_dataset)
    elif selected_operation == "Remove Duplicates":
        session_state.selected_dataset = remove_duplicates(session_state.selected_dataset)
    elif selected_operation == "Handle Categorical Variables":
        session_state.selected_dataset = handle_categorical_variables(session_state.selected_dataset)
    elif selected_operation == "Modify Columns":
        session_state.selected_dataset = modify_columns(session_state.selected_dataset)
    elif selected_operation == "Handle Missing Values":
        session_state.selected_dataset = handle_missing_values(session_state.selected_dataset)
    elif selected_operation == "Handle Datetime Variables":
        session_state.selected_dataset = handle_datetime_variables(session_state.selected_dataset)
    elif selected_operation == "Deal with Skewed Data":
        session_state.selected_dataset = deal_with_skewed_data(session_state.selected_dataset)
    elif selected_operation == "Handle Multicollinearity":
        session_state.selected_dataset = handle_multicollinearity(session_state.selected_dataset)
    elif selected_operation == "Handle Imbalanced Data":
        session_state.selected_dataset = handle_imbalanced_data(session_state.selected_dataset, target_column="your_target_column_name")
    elif selected_operation == "Feature Engineering":
        session_state.selected_dataset = feature_engineering(session_state.selected_dataset)
    elif selected_operation == "Handle Outliers":
        session_state.selected_dataset = handle_outliers(session_state.selected_dataset)
    elif selected_operation == "Standardize/Normalize Data":
        session_state.selected_dataset = standardize_normalize_data(session_state.selected_dataset)
    elif "Drop Selected Rows" in selected_operations:
        session_state.selected_dataset = drop_selected_rows(session_state.selected_dataset)

# Visualization based on user selection
if selected_visualizations:
    st.subheader("Selected Visualizations:")
    for visualization_type in selected_visualizations:
        if visualization_type == "Scatter Plot":
            scatter_plot(session_state.selected_dataset, "x_column", "y_column", color_column="color_column")
        elif visualization_type == "Line Chart":
            line_chart(session_state.selected_dataset, "x_column", "y_column", hue="hue_column")
        elif visualization_type == "Bar Chart":
            bar_chart(session_state.selected_dataset, "x_column", "y_column", hue="hue_column")
        elif visualization_type == "Histogram":
            histogram(session_state.selected_dataset, "column_name", color_hist="blue", bins_hist=10, hue_column_hist=None)
        elif visualization_type == "Box Plot":
            box_plot(session_state.selected_dataset, "x_column", "y_column", hue="hue_column")
        elif visualization_type == "Heatmap":
            heatmap(session_state.selected_dataset)
        elif visualization_type == "Pair Plot":
            pair_plot(session_state.selected_dataset, hue="hue_column")
        elif visualization_type == "Pie Chart":
            pie_chart(session_state.selected_dataset, "column_name")
        elif visualization_type == "Violin Plot":
            violin_plot(session_state.selected_dataset)
        elif visualization_type == "3D Scatter Plot":
            scatter_3d(session_state.selected_dataset, "x_column", "y_column", "z_column", color_column="color_column", size_column="size_column")
        elif visualization_type == "Time Series Plot":
            time_series_plot(session_state.selected_dataset, "date_column", "value_column")
        elif visualization_type == "Geographic Map":
            geo_map(session_state.selected_dataset, "lat_column", "lon_column", size_column="size_column", color_column="color_column", hover_data=["hover_column"])


st.write("---")



#************************************************************************************************************************


# Display the dataset after selected data cleaning operations
st.subheader("Dataset After Data Cleaning Operations:")
st.dataframe(session_state.selected_dataset)

# Create a download button
def download_button(data, file_name, button_text):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert the cleaned dataframe to CSV and encode as base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Add the download button
download_button(session_state.selected_dataset, "cleaned_data", "Download Cleaned Data")

