import cv2
import numpy as np

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
