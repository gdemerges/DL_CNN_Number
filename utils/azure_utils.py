import streamlit as st
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import StringIO

@st.cache_resource
def load_data_from_azure():
    account_name = st.secrets["azure"]["account_name"]
    account_key = st.secrets["azure"]["account_key"]
    container_name = st.secrets["azure"]["container_name"]
    blob_name = st.secrets["azure"]["blob_name"]

    # Connexion à Azure Blob Storage
    blob_service_client = BlobServiceClient(
        account_url=f"https://guiblob.blob.core.windows.net",
        credential=account_key
    )
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    # Téléchargement du fichier CSV depuis Azure Blob Storage
    download_stream = blob_client.download_blob()
    download_content = download_stream.content_as_text()
    test_data = pd.read_csv(StringIO(download_content))

    return test_data
