
import cv2                
import matplotlib.pyplot as plt                        
import logging

# extract pre-trained face detector


def get_face_cascade():
    """
    Load pre-downloaded features
    """
    return cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    """
    Detects a face in the image using cascade features and opencv
    """
    logging.warning(img_path)
    img = cv2.imread(img_path)
    logging.warning(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_cascade()
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0