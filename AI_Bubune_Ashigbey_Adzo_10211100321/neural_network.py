import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def run_neural_network():
    st.title("🧠 Neural Network Classifier")

    uploaded_file = st.file_uploader("📤 Upload a CSV file for classification", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Data Preview:")
        st.dataframe(df.head())

        all_columns = df.columns.tolist()
        target_col = st.selectbox("🎯 Select Target Column (label/class)", all_columns)

        features = [col for col in all_columns if col != target_col]

        X = df[features]
        y = df[target_col]

        # Encode target labels to integers
        le = LabelEncoder()
        y_int = le.fit_transform(y)

        # Normalize numeric inputs
        X = pd.get_dummies(X)  # handle categorical if any
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # One-hot encode for classification
        y_encoded = to_categorical(y_int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        # Hyperparameters
        st.subheader("⚙️ Model Settings")
        epochs = st.slider("Epochs", 1, 100, 20)
        lr = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, format="%.4f")

        # Build model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(y_encoded.shape[1], activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        with st.spinner("Training neural network..."):
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=0)

        # Plot training history
        st.subheader("📉 Training Performance")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train Acc')
        ax[0].plot(history.history['val_accuracy'], label='Val Acc')
        ax[0].set_title("Accuracy")
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Val Loss')
        ax[1].set_title("Loss")
        ax[1].legend()

        st.pyplot(fig)

        # Predict with new input
        st.subheader("🧪 Predict with New Input")
        input_data = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=0.0)
            input_data.append(val)

        input_arr = scaler.transform([input_data])
        prediction = model.predict(input_arr)
        predicted_class = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_class])[0]

        st.success(f"🧠 Predicted Class: {predicted_label}")
