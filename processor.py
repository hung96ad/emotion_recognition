import os
import sys
import logging

import cv2
from keras.models import load_model
from keras.backend import clear_session
import numpy as np
import tensorflow as tf

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './trained_models/emotion_models/fer2013.hdf5'
emotion_labels = get_labels('fer2013')
emotion_offsets = (0, 0)
face_detection = load_detection_model(detection_model_path)
graph = tf.get_default_graph()
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

def process_image(image):
    results = []
    image_array = np.fromstring(image, np.uint8)
    unchanged_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        with graph.as_default():
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]
        results.append(emotion_text)
    return results