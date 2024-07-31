import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Fonction pour télécharger un fichier depuis Azure Blob Storage
def download_blob_to_df(blob_url, account_name, account_key):
    blob_service_client = BlobServiceClient(account_url=f"https://{guiblob}.blob.core.windows.net", credential=account_key)
    blob_client = blob_service_client.get_blob_client(container="guiblob", blob=blob_url)

    blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(BytesIO(blob_data))
    return df

# Charger le fichier CSV depuis Azure Blob Storage
account_name = st.secrets["azure"]["account_name"]
account_key = st.secrets["azure"]["account_key"]
test_blob_url = st.secrets["azure"]["test_blob_url"]

test_data = download_blob_to_df(test_blob_url, account_name, account_key)

model = load_model('mnist_cnn_model.h5')

X_test = test_data.values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

# Fonction pour faire des prédictions
def predict_digit(img, model):
    if len(img.shape) == 3 and img.shape[-1] == 3:  # Vérifier si l'image est en couleur (3 canaux)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[-1] == 4:  # Vérifier si l'image a un canal alpha (4 canaux)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

st.title("Classification de Chiffres avec un CNN")

if 'attempts' not in st.session_state:
    st.session_state['attempts'] = 0
    st.session_state['correct_predictions'] = 0
    st.session_state['incorrect_predictions'] = 0

# Limite des tentatives
MAX_ATTEMPTS = 10

# Première fonctionnalité : Afficher une image aléatoire du dataset de test et prédire
st.header("Tester une image aléatoire du dataset de test")

if st.button("Afficher une image aléatoire"):
    random_idx = np.random.randint(0, X_test.shape[0])
    random_image = X_test[random_idx].reshape(28, 28)
    st.image(random_image, caption="Image aléatoire du dataset", use_column_width=True)

    # Stocker l'index de l'image aléatoire pour l'utiliser dans la prédiction
    st.session_state['random_idx'] = random_idx
    st.session_state['predicted'] = False

if 'random_idx' in st.session_state:
    random_image = X_test[st.session_state['random_idx']].reshape(28, 28, 1)
    if st.button("Predict"):
        random_image = (random_image * 255).astype(np.uint8)  # Assurez-vous que l'image est en entier 8 bits
        digit, confidence = predict_digit(random_image, model)
        st.session_state['prediction'] = digit
        st.session_state['confidence'] = confidence
        st.session_state['predicted'] = True
        st.write(f"Chiffre prédit: {digit} (Confiance: {confidence:.2f})")

if st.session_state.get('predicted', False):
    correct_prediction = st.radio("Le modèle a-t-il correctement classé l'image ?", ("Oui", "Non"))
    if st.button("Valider"):
        if correct_prediction == "Oui":
            st.write("Merci pour votre validation !")
        elif correct_prediction == "Non":
            st.write("Merci pour votre retour. Le modèle continuera à s'améliorer.")
        st.session_state['predicted'] = False

# Deuxième fonctionnalité : Jeu de dessin de chiffres
st.header("Jeu de Dessin de Chiffres")

st.write("Dessinez un chiffre ci-dessous et appuyez sur 'Predict'")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prédiction sur le chiffre dessiné
if st.button("Predict dessin"):
    if canvas_result.image_data is not None:
        img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        digit, confidence = predict_digit(img, model)

        # Mettre à jour le nombre de tentatives
        st.session_state['attempts'] += 1

        st.write(f"Chiffre prédit: {digit} (Confiance: {confidence:.2f})")
        st.session_state['predicted_draw'] = True

if st.session_state.get('predicted_draw', False):
    correct_prediction = st.radio("Le modèle a-t-il correctement classé le dessin ?", ("Oui", "Non"))
    if st.button("Valider dessin"):
        if correct_prediction == "Oui":
            st.session_state['correct_predictions'] += 1
        elif correct_prediction == "Non":
            st.session_state['incorrect_predictions'] += 1

        st.session_state['predicted_draw'] = False

        if st.session_state['attempts'] >= MAX_ATTEMPTS:
            st.write("Vous avez atteint le nombre maximum de tentatives.")
            st.write(f"Prédictions correctes: {st.session_state['correct_predictions']}")
            st.write(f"Prédictions incorrectes: {st.session_state['incorrect_predictions']}")

            # Réinitialiser les tentatives
            if st.button("Recommencer"):
                st.session_state['attempts'] = 0
                st.session_state['correct_predictions'] = 0
                st.session_state['incorrect_predictions'] = 0

# Afficher le nombre de tentatives restantes
st.write(f"Tentatives restantes: {MAX_ATTEMPTS - st.session_state['attempts']}")
