from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.dog_detector import *
from src.breed_detector import *
from src.face_detector import *

app = Flask(__name__)


def detect_face_dog(img_path, model_name='Resnet50'):
    
    if face_detector(img_path):
        return 'A human face is present in the picture'
    
    if dog_detector(img_path):
        label = predict_breed(img_path, model_name)
        return f'A dog with breed {label} is present in the picture'
    
    return 'The picture does not contain a human neither a dog face'


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = detect_face_dog(f'{img_path}')
        
	return render_template("home.html", prediction = p, img_path = img_path)


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()


