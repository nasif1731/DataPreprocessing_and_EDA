import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import google.generativeai as genai

# Initialize Gemini API Client
api_key = "AIzaSyDbVX9RxUYjv6VZJwUqVW9-ZZPyoEL7oVI"  # Replace with actual API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

def generate_features(df):
    # Generate dataset description dynamically
    dataset_info = f"""
    The dataset contains the following columns:
    {df.dtypes.to_string()}

    The goal is to enhance this dataset with meaningful new features.
    Suggest new features based on data types and interactions between columns.
    """
    response = model.generate_content(contents=dataset_info)
    
    return response.text if response else "No features generated."
def convert_numeric_columns(df):
    """Safely convert columns with numeric-like values stored as strings into actual numeric types."""
    possible_numeric_cols = [
        col for col in df.columns 
        if df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all()
    ]
    
    if possible_numeric_cols:
        df[possible_numeric_cols] = df[possible_numeric_cols].astype(float)
    
    return df


st.set_page_config(page_title="Enhanced EDA Tool", layout="wide")

st.title("📊 Exploratory Data Analysis Tool")

# File uploader for CSV and JSON
data_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

if data_file is not None:
    with st.spinner('Loading data...'):
        try:
            if data_file.name.endswith('.csv'):
                df = pd.read_csv(data_file, nrows=1000)
            elif data_file.name.endswith('.json'):
                json_data = json.load(data_file)
                if isinstance(json_data, list):
                    df = pd.json_normalize(json_data)
                elif isinstance(json_data, dict) and "response" in json_data and "data" in json_data["response"]:
                    df = pd.json_normalize(json_data["response"]["data"])
                else:
                    df = pd.json_normalize([json_data])
            df = convert_numeric_columns(df)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df = pd.DataFrame()

    # Convert problematic columns to appropriate types
    if 'apiVersion' in df.columns:
        df['apiVersion'] = df['apiVersion'].astype(str)  # Convert to string

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Basic Information
    st.write("### Dataset Info")
    st.write("Number of Rows:", df.shape[0])
    st.write("Number of Columns:", df.shape[1])
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Statistical Summary
    st.write("### Statistical Summary")
    st.write(df.describe())

    # Numerical and Categorical Column Selection
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Visualization Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📈 Numerical Analysis", "🔤 Categorical Analysis", "🔗 Correlation Analysis", 
         "📊 Advanced Statistics", "🧹 Data Cleaning", "🛠️ Feature Engineering", "📊 Modeling"]
    )

    # Numerical Analysis
    with tab1:
        if not numerical_cols.empty:
            num_col = st.selectbox("Select a numerical column", numerical_cols, help="Choose a column to analyze its distribution.")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[num_col], kde=True, ax=ax)
            st.pyplot(fig)

            # Boxplot for Outlier Detection
            st.write("### Boxplot for Outlier Detection")
            fig_box, ax_box = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[num_col], ax=ax_box)
            st.pyplot(fig_box)
        else:
            st.write("No numerical columns detected.")

    # Categorical Analysis
    with tab2:
        if not categorical_cols.empty:
            cat_col = st.selectbox("Select a categorical column", categorical_cols, help="Choose a column to analyze its distribution.")
            fig = px.bar(df[cat_col].value_counts(), x=df[cat_col].value_counts().index, y=df[cat_col].value_counts().values, labels={'x': cat_col, 'y': 'Count'})
            st.plotly_chart(fig)

            # Pie Chart for Categorical Distribution
            st.write("### Pie Chart for Categorical Distribution")
            fig_pie = px.pie(df, names=cat_col, title=f'Distribution of {cat_col}')
            st.plotly_chart(fig_pie)
        else:
            st.write("No categorical columns detected.")

    # Correlation Analysis
    with tab3:
        if not numerical_cols.empty:
            corr = df[numerical_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            
            st.write("### Pairplot for Visualizing Relationships")
            selected_pairplot_cols = st.multiselect("Select up to 5 numerical columns for pairplot", numerical_cols, default=numerical_cols[:min(5, len(numerical_cols))])
            if len(selected_pairplot_cols) >= 2:
                pairplot_fig = sns.pairplot(df[selected_pairplot_cols])
                st.pyplot(pairplot_fig.fig)
            else:
                st.write("Select at least 2 numerical columns for pairplot.")
        else:
            st.write("No numerical columns detected.")
    # Advanced Statistics
    with tab4:
        st.write("### Advanced Statistical Analysis")
        if not numerical_cols.empty:
            selected_num_col = st.selectbox("Select a numerical column for detailed stats", numerical_cols, help="Choose a column for detailed statistical analysis.")
            st.write("Mean:", df[selected_num_col].mean())
            st.write("Median:", df[selected_num_col].median())
            st.write("Standard Deviation:", df[selected_num_col].std())
            st.write("Skewness:", df[selected_num_col].skew())
            st.write("Kurtosis:", df[selected_num_col].kurtosis())
        else:
            st.write("No numerical columns available for advanced statistics analysis.")

    # Data Cleaning
    with tab5:
        st.write("### Data Cleaning")
        if st.button("Remove Rows with Missing Values"):
            original_shape = df.shape[0]
            df.dropna(inplace=True)
            st.write(f"Removed {original_shape - df.shape[0]} rows with missing values.")
        st.write("### Current Missing Values:")
        st.write(df.isnull().sum())
        
        # Option to fill missing values
        fill_option = st.selectbox("Select a filling method for missing values", ["None", "Mean", "Median", "Mode"])
        if fill_option != "None":
            for col in numerical_cols:
                if fill_option == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif fill_option == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
            st.write("Missing values filled.")

    # Feature Engineering
    with tab6:
        st.write("### Feature Engineering")
        st.write("Enhance your dataset by creating new meaningful features.")
        # Interaction Features
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        interaction_cols = st.multiselect("Select columns for interaction features", numerical_cols)
        
        if st.button("Create Interaction Features"):
            if len(interaction_cols) >= 2:
                for i in range(len(interaction_cols)):
                    for j in range(i + 1, len(interaction_cols)):
                        new_col_name = f"{interaction_cols[i]}_x_{interaction_cols[j]}"
                        df[new_col_name] = df[interaction_cols[i]] * df[interaction_cols[j]]
                st.write("✅ Interaction features created successfully.")
            else:
                st.write("⚠️ Please select at least two numerical columns for interaction features.")

        # Advanced Feature Generation with Gemini API
        st.write("### Advanced Feature Engineering with Gemini API")
        if st.button("Generate Features with Gemini API"):
            st.write("⏳ Calling Gemini API to generate advanced features...")
            try:
                features = generate_features(df)
                st.write("✅ Suggested Features:")
                st.write(features)
            except Exception as e:
                st.write(f"❌ An error occurred: {str(e)}")

        # Display current features
        st.write("### Current Features in Dataset:")
        st.write(df.columns)

    # Modeling
    with tab7:
        st.write("### Regression Modeling")
        target_col = st.selectbox("Select the target variable", numerical_cols)
        
        if target_col in df.columns:
            feature_cols = st.multiselect("Select feature variables", df.columns.drop(target_col))
        else:
            st.error(f"Target column '{target_col}' not found in dataset. Please select a valid column.")
            feature_cols = []
        model_type = st.selectbox("Select Regression Model", ["Linear Regression", "Ridge Regression", "Lasso Regression"])

        if st.button("Train Regression Model"):
            if feature_cols:
                X = df[feature_cols]
                y = df[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Ridge Regression":
                    model = Ridge()
                elif model_type == "Lasso Regression":
                    model = Lasso()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # Plot regression results
                st.write("### Actual vs. Predicted Values")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Regression Model Performance")
                st.pyplot(fig)
            else:
                st.write("Please select at least one feature variable.")

    st.success("Data analysis completed successfully.")
