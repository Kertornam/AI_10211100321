import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

def run_clustering():
    st.title("🧩 Clustering with K-Means")

    st.subheader("📤 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("📄 Dataset Preview:")
        st.dataframe(df.head())

        # Automatically detect numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Dataset must contain at least 2 numeric columns for clustering.")
            return

        st.subheader("🔧 Clustering Settings")
        selected_features = st.multiselect("Select features to use for clustering", numeric_cols, default=numeric_cols[:2])
        n_clusters = st.slider("Number of clusters", 2, 10, 3)

        if len(selected_features) >= 2:
            X = df[selected_features]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)

            df['Cluster'] = cluster_labels

            st.subheader("📊 Clustered Data Preview")
            st.dataframe(df.head())

            st.subheader("📈 Cluster Visualization")
            if len(selected_features) == 2:
                fig = px.scatter(df, x=selected_features[0], y=selected_features[1], color='Cluster', title="2D Cluster Plot")
                st.plotly_chart(fig)
            elif len(selected_features) >= 3:
                fig = px.scatter_3d(df, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                                    color='Cluster', title="3D Cluster Plot")
                st.plotly_chart(fig)

            st.subheader("📥 Download Clustered Dataset")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="clustered_data.csv", mime="text/csv")

        else:
            st.warning("Please select at least two numeric features.")
