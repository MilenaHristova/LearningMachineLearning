import flask
from flask import Flask, render_template, url_for, request, jsonify
from flask_bootstrap import Bootstrap
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from PIL import Image
import keras
#from sklearn.externals import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "Uploads"
Bootstrap(app)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("best_weights_of_old.hdf5")

    loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    imagefile = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(filepath)
    image = Image.open(filepath).convert('RGB')
    return jsonify(prediction=classify_image(loaded_model, image))


def classify_image(model, image):
    classes = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
    im = image.resize((64, 64))
    imArr = numpy.array(im)
    x = numpy.expand_dims(imArr, axis=0)
    x = x / 255.0

    res = model.predict_classes(x)
    keras.backend.clear_session()
    return classes[res[0]]

if __name__ == '__main__':
    app.run(debug=True)

