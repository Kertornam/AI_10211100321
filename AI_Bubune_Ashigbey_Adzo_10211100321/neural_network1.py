import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_neural_network():
    st.title("🧠 Neural Network Classifier (Scikit-learn MLP)")

    # Step 1: Upload dataset
    uploaded_file = st.file_uploader("📁 Upload a CSV file for classification", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Preview", df.head())

        # Step 2: Select the target column
        target_column = st.selectbox("🎯 Select the target column (label)", df.columns)

        # Step 3: Prepare features (X) and label (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode labels if they are not numeric
        if y.dtype == object or y.dtype == 'bool':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Use only numeric features
        X = X.select_dtypes(include=["int64", "float64"])

        if X.empty:
            st.error("❌ No numeric features found. Please upload a dataset with numeric columns.")
            return

        # Step 4: Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 5: Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Step 6: Sidebar - Hyperparameters
        st.sidebar.header("⚙️ Neural Network Settings")
        hidden_layer_size = st.sidebar.slider("Hidden layer size", 10, 200, 100)
        max_iter = st.sidebar.slider("Max iterations", 100, 1000, 300)
        learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01, step=0.001)

        # Step 7: Train MLP model
        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,),
                              max_iter=max_iter,
                              learning_rate_init=learning_rate,
                              random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Step 8: Show results
        acc = accuracy_score(y_test, y_pred)
        st.subheader("📊 Model Performance")
        st.write(f"✅ Accuracy: **{acc:.2f}**")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, zero_division=0))
