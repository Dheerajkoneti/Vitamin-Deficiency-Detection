import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

CLASS_NAMES = ["Normal", "Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D"]

# Load models
cnn = load_model("models/cnn_model.h5")
vgg = load_model("models/vgg16_model.h5")
resnet = load_model("models/resnet50_model.h5")
mobilenet = load_model("models/mobilenet_model.h5")

models = [cnn, vgg, resnet, mobilenet]

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def extract_regions(img):
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    def crop(points):
        xs = [int(lm[p].x * w) for p in points]
        ys = [int(lm[p].y * h) for p in points]
        return img[min(ys):max(ys), min(xs):max(xs)]

    return {
        "eyes": crop([33,133,362,263]),
        "lips": crop([61,291,0,17]),
        "skin": crop([234,454,10,152])
    }

def predict_region(img, models):
    img = cv2.resize(img,(224,224))/255.0
    img = np.expand_dims(img,0)
    preds = [m.predict(img,verbose=0)[0] for m in models]
    avg = np.mean(preds,axis=0)
    idx = np.argmax(avg)
    return CLASS_NAMES[idx], round(float(avg[idx]*100),2)

# Webcam loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    regions = extract_regions(frame)

    if regions:
        y = 30
        for name, region in regions.items():
            label, conf = predict_region(region, models)
            cv2.putText(frame, f"{name}: {label} ({conf}%)",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)
            y += 30

    cv2.imshow("Vitamin Deficiency Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()