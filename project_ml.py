import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import plotly.express as px  # Required import
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize


# Initialize session state
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = None


if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model_name' not in st.session_state:
    st.session_state.trained_model_name = None
if 'model_category' not in st.session_state:
    st.session_state.model_category = None
if 'model_result' not in st.session_state:
    st.session_state.model_result = None



if 'model_evaluation' not in st.session_state:
    st.session_state.model_evaluation = None
if 'model_evaluation_result' not in st.session_state:
    st.session_state.model_evaluation_result = None
if 'model_evaluation_note' not in st.session_state:
    st.session_state.model_evaluation_note = None


if 'model_evaluation_confusion_matrix' not in st.session_state:
    st.session_state.model_evaluation_confusion_matrix = None
if 'model_evaluation_accuracy' not in st.session_state:
    st.session_state.model_evaluation_accuracy = None
if 'model_evaluation_f1_score' not in st.session_state:
    st.session_state.model_evaluation_f1_score = None

def upload_page():
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_data = df
            st.session_state.processed_data = df.copy()
            
            # Save data info in session
            data_info = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'numeric_columns': len(df.select_dtypes(include=np.number).columns),
                'categorical_columns': len(df.select_dtypes(include='object').columns),
                'columns_info': df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Type'})
            }
            st.session_state.data_info = data_info
            
            st.success("Data uploaded successfully!")
            
            display_data_summary(df, data_info)
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
    else:
        if st.session_state.get('data_info'):
            display_data_summary(st.session_state.raw_data, st.session_state.data_info)

def display_data_summary(df, data_info):
    st.subheader("Data Summary")
    
    # Create three columns layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write("**Data Preview:**")
        st.dataframe(df.head())
    
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"Rows: {data_info['rows']}")
        st.write(f"Columns: {data_info['columns']}")
        st.write(f"Numerical: {data_info['numeric_columns']}")
        st.write(f"Categorical: {data_info['categorical_columns']}")
    
    with col3:
        st.write("**Columns & Types:**")
        # Format the types for better display
        columns_info = data_info['columns_info'].copy()
        columns_info['Type'] = columns_info['Type'].astype(str)
        st.dataframe(columns_info, height=200, hide_index=True)

def preprocessing_page():
    if st.session_state.processed_data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("🔧 Data Preprocessing")
    
    # Create sidebar menu for preprocessing options
    st.sidebar.subheader("Preprocessing Options")
    option = st.sidebar.radio(
        "Choose Preprocessing Type",
        ["Missing Values", "Encoding", "Outliers", "Normalization"]
    )
    
    if option == "Missing Values":
        handle_missing_values()
    elif option == "Encoding":
        handle_categorical_encoding()
    elif option == "Outliers":
        handle_outliers()
    elif option == "Normalization":
        handle_normalization()

def handle_missing_values():

    st.subheader("🔄 Missing Values Handling")

    data = st.session_state.processed_data.copy()

    missing_cols = data.columns[data.isna().any()].tolist()
    
    if not missing_cols:
        st.success("✅ No missing values found!")
        return
    
    
    missing_percent = {col: (data[col].isna().sum() / len(data)) * 100
                        for col in missing_cols}
    
    st.write("**Columns with missing values:**")
    missing_info = pd.DataFrame({
        'Column': missing_cols,
        'Missing Count': [data[col].isna().sum() for col in missing_cols],
        'Missing %': [f"{missing_percent[col]:.2f}%" for col in missing_cols],
        'Data Type': [data[col].dtype for col in missing_cols]
    })
    st.dataframe(missing_info)
    
    st.markdown("---")
    st.markdown("### 🚨 High Missing Percentage Columns (>50%)")
    
    high_missing_cols = [col for col in missing_cols if missing_percent[col] > 50]
    
    if high_missing_cols:
        st.warning(f"Found {len(high_missing_cols)} columns with >50% missing values:")
        st.write(high_missing_cols)
        
        if st.checkbox("Drop all columns with >50% missing values", key="drop_high_missing"):
            data.drop(columns=high_missing_cols, inplace=True)
            st.success(f"Dropped {len(high_missing_cols)} columns with high missing values!")
            st.session_state.processed_data = data
            return
    else:
        st.info("No columns with >50% missing values found")
    
    
    missing_cols = data.columns[data.isna().any()].tolist()
    if not missing_cols:
        return
    
    
    st.markdown("---")
    st.markdown("### 🛠️ Simple Imputation")
    simple_cols = st.multiselect(
        "Select columns for basic imputation",
        missing_cols
    )
    
    if simple_cols:
       
        numeric_cols = [col for col in simple_cols if data[col].dtype in ['int64', 'float64']]
       
        categorical_cols = [col for col in simple_cols if data[col].dtype not in ['int64', 'float64']]

        if numeric_cols:
            st.write("**Numeric Columns Options:**")
            num_method = st.selectbox(
                "Method for numeric columns:",
                ["Mean", "Median", "Constant", "Drop Rows"],
                key="num_simple"
            )
            if num_method == "Constant":
                num_const = st.text_input("Enter value:", "0", key="num_const")
        
        if categorical_cols:
            st.write("**Categorical Columns Options:**")
            cat_method = st.selectbox(
                "Method for categorical columns:",
                ["Most Frequent", "Constant", "Drop Rows"],
                key="cat_simple"
            )
            if cat_method == "Constant":
                cat_const = st.text_input("Enter value:", "Missing", key="cat_const")
        
        if st.button("Apply Simple Imputation"):
            try:
                if numeric_cols:
                    if num_method == "Mean":
                        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
                    elif num_method == "Median":
                        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                    elif num_method == "Constant":
                        try:
                            const = float(num_const)
                            data[numeric_cols] = data[numeric_cols].fillna(const)
                        except:
                            data[numeric_cols] = data[numeric_cols].fillna(num_const)
                    elif num_method == "Drop Rows":
                        data.dropna(subset=numeric_cols, inplace=True)
                
                
                if categorical_cols:
                    if cat_method == "Most Frequent":
                        for col in categorical_cols:
                            data[col] = data[col].fillna(data[col].mode()[0])
                    elif cat_method == "Constant":
                        data[categorical_cols] = data[categorical_cols].fillna(cat_const)
                   
                    elif cat_method == "Drop Rows":
                        data.dropna(subset=categorical_cols, inplace=True)
                
                st.success("Simple imputation applied!")
                st.session_state.processed_data = data
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    
    st.markdown("---")
    st.markdown("### ⚙️ Advanced Imputation (Numeric Only)")
    numeric_missing = [col for col in missing_cols if pd.api.types.is_numeric_dtype(data[col])]
     #numeric_missing= [col for col in simple_cols if data[col].dtype in ['int64', 'float64']]
       
    if numeric_missing:
        advanced_cols = st.multiselect(
            "Select numeric columns for advanced imputation",
            numeric_missing
        )
        
        if advanced_cols:
            method = st.radio(
                "Select method:",
                ["KNN Imputer", "Iterative Imputer"],
                horizontal=True
            )
            
            if method == "KNN Imputer":
                n_neighbors = st.slider("Number of neighbors", 2, 10, 5)
            else:
                max_iter = st.slider("Max iterations", 1, 20, 10)
                #sample_posterior = st.checkbox("Sample from posterior", True)
            
            if st.button("Apply Advanced Imputation"):
                try:
                    if method == "KNN Imputer":
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                    else:
                        imputer = IterativeImputer(max_iter=max_iter, 
                                               
                                                random_state=42)
                    
                    data[advanced_cols] = imputer.fit_transform(data[advanced_cols])
                    st.success(f"{method} applied successfully!")
                    st.session_state.processed_data = data
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("No numeric columns with missing values for advanced imputation")
    
    # Display results
    st.markdown("---")
    st.subheader("Data After Imputation")
    st.dataframe(data.head())
    
    if st.button("💾 Save Processed Data"):
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv", 
            mime="text/csv"
        )

def handle_categorical_encoding():
    st.subheader("🔡 Categorical Variables Encoding")
    data = st.session_state.processed_data.copy()

    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.success("✅ No categorical variables found!")
        return data
    
    st.write("**Categorical Variables:**")
    cat_info = pd.DataFrame({
        'Column': categorical_cols,
        'Unique Values': [data[col].nunique() for col in categorical_cols],
        'Sample Values': [data[col].unique()[:3] for col in categorical_cols]
    })
    st.dataframe(cat_info)
 
    st.markdown("---")
    st.markdown("### 🛠️ Encoding Options")
    
   
    cols_to_encode = st.multiselect(
        "Select columns to encode",
        categorical_cols
    )
    
    if cols_to_encode:
        
        encoding_method = st.radio(
            "Select encoding method:",
            ["Label Encoding", "One-Hot Encoding", "Binary Encoding"],
            horizontal=True
        )
        
      
        if st.button("Apply Encoding"):
            try:
                if encoding_method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    for col in cols_to_encode:
                        data[col] = le.fit_transform(data[col])
                    st.success("Label Encoding applied!")
                
                elif encoding_method == "One-Hot Encoding":
                    data = pd.get_dummies(data, columns=cols_to_encode, drop_first=False)
                    st.success("One-Hot Encoding applied!")
                
                elif encoding_method == "Binary Encoding":
                    import category_encoders as ce
                    encoder = ce.BinaryEncoder(cols=cols_to_encode)
                    data = encoder.fit_transform(data)
                    st.success("Binary Encoding applied!")
                
                st.session_state.processed_data = data
                return data
                
            except Exception as e:
                st.error(f"Error during encoding: {str(e)}")
                return data
    
    # Display results
    st.markdown("---")
    st.subheader("Data After Encoding")
    st.dataframe(data.head())
    
    if st.button("💾 Save Encoded Data"):
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="encoded_data.csv",
            mime="text/csv"
        )
    
    return data

def handle_outliers():
    st.subheader("📊 Outlier Detection and Treatment")
    data = st.session_state.processed_data.copy()

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for outlier detection!")
        return data

    selected_col = st.selectbox("Select column to analyze for outliers:", numeric_cols)

    detection_method = st.radio("Select detection method:", ["IQR Method", "Z-Score", "Box Plot Visualization"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{data[selected_col].mean():.2f}")
    with col2:
        st.metric("Median", f"{data[selected_col].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{data[selected_col].std():.2f}")

    outliers = None
    if detection_method == "IQR Method":
        Q1 = data[selected_col].quantile(0.25)
        Q3 = data[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]

    elif detection_method == "Z-Score":
        threshold = st.slider("Z-Score threshold", 2.0, 5.0, 3.0, 0.5)
        z_scores = (data[selected_col] - data[selected_col].mean()) / data[selected_col].std()
        outliers = data[np.abs(z_scores) > threshold]

    elif detection_method == "Box Plot Visualization":
        fig = px.box(data, y=selected_col, title=f"Box Plot of {selected_col}")
        st.plotly_chart(fig)
        return data

    if outliers is not None and not outliers.empty:
        st.warning(f"Found {len(outliers)} outliers in {selected_col}")
        st.dataframe(outliers.head())

        treatment_method = st.radio("Select treatment method:", ["Delete Outliers", "Clip Values"], horizontal=True)

        if st.button("Apply Treatment"):
            try:
                if treatment_method == "Delete Outliers":
                    if detection_method == "IQR Method":
                        data = data[(data[selected_col] >= lower_bound) & (data[selected_col] <= upper_bound)]
                    else:  # Z-Score
                        data = data[np.abs(z_scores) <= threshold]
                    st.success(f"Deleted {len(outliers)} outliers")

                elif treatment_method == "Clip Values":
                    if detection_method == "IQR Method":
                        data[selected_col] = data[selected_col].clip(lower_bound, upper_bound)
                    else:  # Z-Score
                        mean = data[selected_col].mean()
                        std = data[selected_col].std()
                        data[selected_col] = data[selected_col].clip(mean - threshold*std, mean + threshold*std)
                    st.success("Clipped values to detection bounds")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before treatment**")
                    st.dataframe(outliers.head())
                with col2:
                    st.write("**After treatment**")
                    if treatment_method == "Delete Outliers":
                        st.info("Outliers removed from dataset")
                    else:
                        st.dataframe(data.loc[outliers.index].head())

                st.session_state.processed_data = data
                st.success("Outlier treatment applied!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif outliers is not None:
        st.success("No outliers detected with current method")

    st.markdown("---")
    st.subheader("🌐 Winsorization (Manual)")

    selected_col = st.selectbox("Select column for Winsorization:", numeric_cols, key="winsor_col")
    lower_limit = st.slider("Lower limit (%)", 0, 100, 5)
    upper_limit = st.slider("Upper limit (%)", 0, 100, 5)

    if st.button("Apply Winsorization"):
        try:
            before = data[[selected_col]].copy()
            data[selected_col] = winsorize(data[selected_col], limits=(lower_limit/100, upper_limit/100))
            after = data[[selected_col]]

            col1, col2 = st.columns(2)
            with col1:
                st.write("Before treatment")
                st.dataframe(before.head())
            with col2:
                st.write("After treatment")
                st.dataframe(after.head())

            st.session_state.processed_data = data
            st.success(f"Winsorized {selected_col} - Lower {lower_limit:.0f}%, Upper {upper_limit*100:.0f}%")
        except Exception as e:
            st.error(f"Error during Winsorization: {e}")

    return data

def handle_normalization():
    st.subheader("📏 Data Normalization and Transformation")
    data = st.session_state.processed_data.copy()
   
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    selected_cols = st.multiselect(
        "Select columns for normalization:",
        options=numeric_cols,
        default=numeric_cols
    )
    
    if not selected_cols:
        st.warning("Please select at least one column for normalization.")
        return

    normalization_method = st.selectbox(
        "Select normalization method:",
        ("None", "Min-Max Scaling", "Standard Scaling")
    )

    if normalization_method != "None":
        st.info(f"Applying {normalization_method} on selected columns...")
        if normalization_method == "Min-Max Scaling":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaled_data = scaler.fit_transform(data[selected_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=selected_cols)

        st.success(f"{normalization_method} Applied Successfully!")
        st.dataframe(scaled_df)

        data[selected_cols] = scaled_df
        st.session_state.processed_data = data
   
    st.subheader("⚡ Skewness Analysis")

    skew_eligible_cols = [col for col in numeric_cols if data[col].nunique() > 1]

    if skew_eligible_cols:
        feature = st.selectbox("Select feature for Skewness Analysis:", skew_eligible_cols)

        skewness_before = data[feature].skew()
        st.write(f"Skewness of {feature}: {skewness_before:.4f}")

        transformation = st.selectbox(
            "Select transformation to reduce skewness:",
            ("None", "Log Transformation", "Box-Cox Transformation", "Yeo-Johnson Transformation")
        )

        if transformation == "Log Transformation":
            st.info("Applying Log Transformation...")

            if (data[feature] <= 0).any():
                st.error("Log Transformation is not applicable: the feature contains zero or negative values.")
            else:
                data[f"log_{feature}"] = np.log(data[feature] + 1e-6)
                st.dataframe(data[[f"log_{feature}"]])
                st.write(f"New Skewness after Log: {data[f'log_{feature}'].skew():.4f}")

        elif transformation == "Box-Cox Transformation":
            st.info("Applying Box-Cox Transformation...")

            if (data[feature] <= 0).any():
                st.error("Box-Cox Transformation is not applicable: the feature must contain only positive values.")
            else:
                try:
                    transformed_data, _ = boxcox(data[feature])
                    data[f"boxcox_{feature}"] = transformed_data
                    st.dataframe(data[[f"boxcox_{feature}"]])
                    st.write(f"New Skewness after Box-Cox: {pd.Series(transformed_data).skew():.4f}")
                except Exception as e:
                    st.error(f"Box-Cox Transformation failed: {e}")

        elif transformation == "Yeo-Johnson Transformation":
            st.info("Applying Yeo-Johnson Transformation...")
            try:
                pt = PowerTransformer(method='yeo-johnson')
                transformed_data = pt.fit_transform(data[[feature]])
                data[f"yeojohnson_{feature}"] = transformed_data
                st.dataframe(data[[f"yeojohnson_{feature}"]])
                st.write(f"New Skewness after Yeo-Johnson: {pd.Series(transformed_data.flatten()).skew():.4f}")
            except Exception as e:
                st.error(f"Yeo-Johnson Transformation failed: {e}")

        st.session_state.processed_data = data

    else:
        st.warning("No suitable numeric columns found for skewness analysis.")




def visualization_page():
    if st.session_state.processed_data is None:
        st.warning("Please upload data first!")
        return
    
    data = st.session_state.processed_data.copy()
    
    st.subheader("📊 Data Visualization Section")

    st.info("Select a visualization type and configure the options below.")

    visualization_type = st.selectbox(
        "Choose Visualization Type:",
        [
            "Histogram",
            "Boxplot",
            "Scatter Plot (Seaborn)",
            "Regression Plot",
            "Bar Plot",
            "Line Plot",
            "Scatter Plot (Matplotlib)"
        ]
    )

    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if visualization_type == "Histogram":
        selected_column = st.selectbox("Select Column for Histogram:", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(data=data, x=selected_column, kde=True, ax=ax)
        ax.set_title(f'Histogram of {selected_column}')
        st.pyplot(fig)

    elif visualization_type == "Boxplot":
        selected_column = st.selectbox("Select Column for Boxplot:", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=selected_column, ax=ax)
        ax.set_title(f'Boxplot of {selected_column}')
        st.pyplot(fig)

    elif visualization_type == "Scatter Plot (Seaborn)":
        x_axis = st.selectbox("Select X-axis (Numeric):", numeric_cols)
        y_axis = st.selectbox("Select Y-axis (Numeric):", numeric_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
        ax.set_title(f'Scatter Plot of {x_axis} vs {y_axis}')
        st.pyplot(fig)

    elif visualization_type == "Regression Plot":
        x_axis = st.selectbox("Select X-axis (Numeric) for Regression:", numeric_cols)
        y_axis = st.selectbox("Select Y-axis (Numeric) for Regression:", numeric_cols)
        fig = sns.lmplot(x=x_axis, y=y_axis, data=data, aspect=1.5)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'Regression Plot of {x_axis} vs {y_axis}')
        st.pyplot(fig.figure)

    elif visualization_type == "Bar Plot":
        y_axis = st.selectbox("Select Y-axis (Numeric):", numeric_cols)
        x_axis = st.selectbox("Select X-axis (Categorical):", categorical_cols)
       

        fig, ax = plt.subplots()

        sns.barplot(data= data, x=x_axis, y=y_axis,  palette="spring", ax=ax)
        
        ax.set_title(f'Bar Plot of {x_axis} by {y_axis}')
        st.pyplot(fig)

    elif visualization_type == "Line Plot":
        x_axis = st.selectbox("Select X-axis (Categorical or Numeric):", data.columns.tolist())
        y_axis = st.selectbox("Select Y-axis (Numeric):", numeric_cols)
        fig, ax = plt.subplots()
        plt.plot(data[x_axis], data[y_axis], marker='o')
        plt.title(f'Line Plot of {y_axis} over {x_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        st.pyplot(fig)

    elif visualization_type == "Scatter Plot (Matplotlib)":
        x_axis = st.selectbox("Select X-axis (Numeric):", numeric_cols)
        y_axis = st.selectbox("Select Y-axis (Numeric):", numeric_cols)
        fig, ax = plt.subplots()
        plt.scatter(data[x_axis], data[y_axis])
        plt.title(f'Scatter Plot of {x_axis} vs {y_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        st.pyplot(fig)

def model_section():
    st.subheader("🧠 Model Training Section")
    st.info("Explore and train machine learning models by category.")

  
    regression_models = [
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "K-Nearest Neighbors Regressor",
        "Linear Regression"
        
    ]

    classification_models = [
        "Logistic Regression",
        "Naive Bayes",
        "Support Vector Machine (SVM)",
        "Decision Tree Classifier",
        "Random Forest Classifier",
    
        "K-Nearest Neighbors Classifier"
    ]

    clustering_models = [
        "K-Means",
        "DBSCAN",
        "Agglomerative Clustering",
        
    ]

    def render_model(model_name, category):
       st.markdown(f"#### ✅ {model_name}")
       data = st.session_state.processed_data.copy()

       if data is not None:
          numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
          if category == "Regression" and model_name in ["K-Nearest Neighbors Regressor", "Decision Tree Regressor", "Random Forest Regressor", "Linear Regression"]:
             
            target_col = st.selectbox("Select target column (Y):", numeric_cols, key=f"{model_name}_target")
             
            available_features = [col for col in numeric_cols if col != target_col]
            default_features = available_features  # Set default to all available features

        
            feature_cols = st.multiselect("Select feature columns (X):",  available_features, default=default_features  ,key=f"{model_name}_features")

            if st.button(f"Run {model_name}", key=f"{category}_{model_name}"):
                 if feature_cols and target_col:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                    if model_name == "K-Nearest Neighbors Regressor":
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor(n_neighbors=5)
                    elif model_name == "Decision Tree Regressor":
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor(random_state=42)
                    elif model_name == "Random Forest Regressor":
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(random_state=42, n_estimators=10)
                    elif model_name == "Linear Regression":
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                    X = data[feature_cols]
                    y = data[target_col]

                    if X.isna().any().any() or y.isna().any():
                        st.error("Data contains missing values. Please handle missing values before training the model.")
                        return

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    accuracy = model.score(X_test, y_test)
                    error_rate = 1 - accuracy

                    st.success("Model trained successfully.")

                    comparison_df = pd.DataFrame({
                        "Actual": y_test.values,
                        "Predicted": y_pred
                    })

                    st.subheader("📋 Target Prediction Comparison")
                    st.dataframe(comparison_df.head(20))
                    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}))

                    st.session_state["model_trained"] = True
                    st.session_state["trained_model_name"] = model_name
                    st.session_state["model_category"] = category
                    st.session_state["model_result"] = {
                        "r2_score": round(r2, 3),
                        "mae": round(mae, 3),
                        "mse": round(mse, 3),
                        "accuracy": round(accuracy, 3),
                        "error_rate": round(error_rate, 3),
                        "note": f"{model_name} trained on {len(feature_cols)} features."
                    }
                 else:
                    st.warning("Please select target and feature columns first.")
          else:
            st.button(f"Run {model_name}", key=f"{category}_{model_name}")
            st.info("This model is not yet implemented.")
       else:
        st.warning("Please upload and preprocess data first.")

    # ========== واجهة عرض الموديلات ==========
    with st.expander("🔵 Regression Models"):
        for model in regression_models:
            render_model(model, "Regression")

    with st.expander("🟢 Classification Models"):
        for model in classification_models:
            render_model(model, "Classification")

    with st.expander("🟣 Clustering Models"):
        for model in clustering_models:
            render_model(model, "Clustering")


def model_evaluation():
    st.subheader("📈 Model Evaluation")

    if "model_trained" not in st.session_state or not st.session_state["model_trained"]:
        st.warning("Please train a model first from the 'Model Section'.")
        return

    model_name = st.session_state["trained_model_name"]
    category = st.session_state["model_category"]
    results = st.session_state["model_result"]

    st.success(f"Model: {model_name} ({category})")

    # ========== عرض النتائج حسب نوع الموديل ==========
    st.markdown("### Evaluation Results")

    if category == "Regression":
        # عرض مقاييس ريجرشن افتراضية (placeholder)
        st.metric("R² Score", results.get("r2_score","N/A"))
        st.metric("MAE", results.get("mae","N/A"))
        st.metric("MSE", results.get("mse", "N/A"))
        st.metric("Accuracy", results.get("accuracy","N/A"))
        st.metric("Error Rate", results.get("error_rate", "N/A"))
    elif category == "Classification":
        # عرض مقاييس classification افتراضية (placeholder)
        st.metric("Accuracy", results.get("accuracy","N/A"))
        st.metric("F1 Score", results.get("f1_score", "N/A"))
        st.write("Confusion Matrix:")
        st.write(results.get("confusion_matrix", [[50, 3], [5, 42]]))

    elif category == "Clustering":
        # عرض مقياس clustering افتراضي (placeholder)
        st.metric("Silhouette Score", results.get("silhouette_score","N/A"))

    # ملاحظة أو وصف إضافي
    if "note" in results:
        st.info(results["note"])

def main():
    st.sidebar.title("Main Menu")
    page = st.sidebar.radio(
        "Go to",
        ["Upload Data", "Data Preprocessing", "Data Visualization", "Model Selection", "Model Evaluation"]
    )
    
    if page == "Upload Data":
        upload_page()
    elif page == "Data Preprocessing":
        preprocessing_page()
    elif page == "Data Visualization":
        visualization_page()
    elif page == "Model Selection":
        model_section()
    elif page == "Model Evaluation":
        model_evaluation()

if __name__ == "__main__":
    main()