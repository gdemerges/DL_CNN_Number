import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

model = load_model('mnist_cnn_model.h5')

test_data = pd.read_csv('data/test.csv')

X_test = test_data.values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

def predict_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction)

st.title("Classification de Chiffres avec un CNN")

st.write("Dessinez un chiffre ou sélectionnez une image aléatoire du dataset")

# Option pour dessiner un chiffre
draw_option = st.checkbox("Dessiner un chiffre")

if draw_option:
    st.write("Dessinez un chiffre ci-dessous et appuyez sur 'Classifier'")
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

    if st.button("Classifier"):
        if canvas_result.image_data is not None:
            img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))
            img = img.reshape(1, 28, 28, 1) / 255.0

            prediction = model.predict(img)
            st.write(f"Chiffre prédit: {np.argmax(prediction)}")

# Option pour sélectionner une image aléatoire du dataset
if st.button("Classer une image aléatoire du dataset"):
    random_idx = np.random.randint(0, X_test.shape[0])
    random_image = X_test[random_idx].reshape(28, 28)

    st.image(random_image, caption="Image aléatoire du dataset", use_column_width=True)

    prediction = model.predict(random_image.reshape(1, 28, 28, 1))
    st.write(f"Chiffre prédit: {np.argmax(prediction)}")

st.title("Reconnaissance de chiffres via la webcam")

run = st.checkbox('Lancer la webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Erreur : Impossible de capturer l'image")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    prediction = predict_image(image)

    cv2.putText(frame_rgb, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(frame_rgb)

cap.release()
