import streamlit as st
from tensorflow.keras.models import load_model
from azure_utils import load_data_from_azure

@st.cache_resource
def load_model_and_data():
    test_data = load_data_from_azure()

    model = load_model('model/mnist_cnn_model.h5')

    X_test = test_data.values / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_test, model
