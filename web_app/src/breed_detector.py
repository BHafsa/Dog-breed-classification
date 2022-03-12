import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from .extract_bottleneck_features import *
from .dog_detector import path_to_tensor
import pickle


def get_model(train_shape):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_shape))
    model.add(Dense(133, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def load_model(model_name):
    # this works only for resnet50
    model = get_model((7, 7, 2048))
    model.load_weights(f'data/weights.best.{model_name}.hdf5')
    return model
    
def extract_bottleneck(model_name, tensor):
    # the name of the model 
    extractors = {
        'VGG16': extract_VGG16,
        'VGG19': extract_VGG19,
        'Resnet50': extract_Resnet50,
        'Xception': extract_Xception,
        'InceptionV3': extract_InceptionV3,
    }
    
    return extractors[model_name](tensor)
    
def predict_breed(img_path, model_name="Resnet50"):
    # extract bottleneck features
    bottleneck_feature = extract_bottleneck(model_name, path_to_tensor(img_path))
    # obtain predicted vector
    model= load_model(model_name)
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    with open('data/dog_names.pkl', 'rb') as f:
        dog_names = pickle.load(f)
        
    return dog_names[np.argmax(predicted_vector)]