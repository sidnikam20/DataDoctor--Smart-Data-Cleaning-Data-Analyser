

# app.py

import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
from io import BytesIO
from data_cleaner import basic_data_cleaning, standardize_numeric_columns
import plotly.express as px

st.title("🧹Digital Data Analyser & Smart Data Cleaning App")
with st.sidebar:
    st.markdown("## 👩‍💻 About the Creator")
    st.markdown("""
**Name:** Siddhi Nikam  
**Role:** B.Tech (Computer Engineering & Artificial intelligence)  
**College:** G H Raisoni College of Engineering and Management, Pune  

[🔗 LinkedIn](https://www.linkedin.com/in/siddhi-nikam-963a78251/)  
[📧 Mail](mailto:sidnikam2004@gmail.com)

🚀 OPEN TO LEARN AND BUILD WITH TECHNOLOGY!
""")
    st.markdown("💼OPEN TO WORK! Ready to contribute my skills to build even better technology 🔍💡🔧")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if "cleaned_df" not in st.session_state:
  st.session_state.cleaned_df = None
if "encoded_df" not in st.session_state:
    st.session_state.encoded_df = None


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,encoding='latin')
    st.write("📊 Raw Dataset Preview:")
    st.dataframe(df.head())

    st.markdown("---")

    # Column drop selection
    st.subheader("❓ Columns to Drop(optional)")
    drop_cols = st.multiselect("Select columns you want to remove", options=df.columns.tolist())

    # Missing value handling
    st.subheader("🛠 Missing Value Strategy")
    strategy = st.radio(
        "How should missing values be handled?",
        options=["drop", "mean", "median"],
        horizontal=True
    )

    # Trigger cleaning
    if st.button("🚀 Clean My Data"):
        cleaned_df = basic_data_cleaning(df, missing_strategy=strategy, columns_to_drop=drop_cols)
        if cleaned_df.empty or cleaned_df.shape[1] == 0:
            st.error("❌ Cleaning resulted in an empty dataset. Please revise your column selections.")
            st.stop()
        st.session_state.cleaned_df = cleaned_df

        st.success("✅ Data cleaned successfully!")
        st.write("🔍 Cleaned Dataset Preview:")
        st.dataframe(cleaned_df.head())

    # Download link

        @st.cache_data
        def convert_df(df):
          return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(cleaned_df)
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv',
            key="download_cleaned"
        )
# Auto EDA section
if st.session_state.cleaned_df is not None:
    st.markdown("---")
    st.header("🔍 Automated Exploratory Data Analysis")

    df = st.session_state.cleaned_df

    st.subheader("🧾 Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {df.columns.tolist()}")
    st.write("**Missing Values (per column):**")
    st.write(df.isna().sum())

    st.subheader("📊 Descriptive Statistics")
    st.dataframe(df.describe(include='all').T)

    # Correlation Heatmap
    import plotly.express as px
    import plotly.graph_objects as go

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if not numeric_df.empty:
        st.subheader("📈 Interactive Correlation Heatmap")
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("📉 Interactive Feature Distribution")
    selected_column = st.selectbox("Choose column for histogram", numeric_df.columns, key="hist")

    if selected_column:
        fig = px.histogram(df, x=selected_column, marginal="box", nbins=30, title=f"Distribution of {selected_column}")
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("🧪 Outlier Detection (Boxplot)")
    outlier_col = st.selectbox("Select column for boxplot", numeric_df.columns, key="box")

    if outlier_col:
        fig = px.box(df, y=outlier_col, points="all", title=f"Boxplot of {outlier_col}")
        st.plotly_chart(fig, use_container_width=True)
    if st.checkbox("📏 Standardize numeric columns"):
        st.session_state.cleaned_df = standardize_numeric_columns(st.session_state.cleaned_df)
        st.success("✅ Dataset standardized successfully!")
        st.subheader("🔍 Standardized Dataset Preview")
        st.dataframe(st.session_state.cleaned_df.head())

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(st.session_state.cleaned_df)
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name='final_cleaned_data.csv',
        mime='text/csv',
        key="download_standardized"
    )
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    st.markdown("---")

    st.header("🔢 Encode Categorical Variables")
    df = st.session_state.cleaned_df.copy()

    # Detect categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(cat_cols) == 0:
        st.info("✅ No categorical columns detected to encode.")
        st.session_state.numeric_df = df
    else:
        st.write("🧠 Categorical Columns Detected:", cat_cols)

        encoding_method = st.radio(
            "Choose Encoding Method:",
            options=["One-Hot Encoding", "Label Encoding"],
            horizontal=True
        )

        if st.button("🚀 Encode Now"):
            encoded_df = df.copy()

            if encoding_method == "One-Hot Encoding":
                encoded_df = pd.get_dummies(encoded_df, columns=cat_cols, drop_first=True)
            else:
                label_encoder = LabelEncoder()
                for col in cat_cols:
                    encoded_df[col] = label_encoder.fit_transform(encoded_df[col].astype(str))

            st.session_state.encoded_df = encoded_df
            st.success("✅ Encoding completed!")
            st.dataframe(encoded_df.head())
            csv = encoded_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Encoded Data", data=csv, file_name="encoded_data.csv", mime="text/csv",key='encoded_download')

            from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if st.session_state.encoded_df is not None:
    st.markdown("---")
    st.header("🤖 Train a Machine Learning Model")

    df = st.session_state.encoded_df

    all_columns = df.columns.tolist()

    # Feature and target selection
    target = st.selectbox("🎯 Select Target Variable", options=all_columns)

    features = st.multiselect("🧬 Select Feature Columns", options=[col for col in all_columns if col != target])

    if features and target:
        X = df[features]
        y = df[target]

        test_size = st.slider("📊 Test Set Size (in %)", 10, 50, 20)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        model_choice = st.radio("🧠 Choose Model", ["Random Forest", "Logistic Regression"], horizontal=True)

        if st.button("🚀 Train Model"):
            if model_choice == "Random Forest":
                model = RandomForestClassifier()
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained! Accuracy: **{acc:.2f}**")

            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("📌 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

    

