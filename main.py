from flask import Flask, render_template, redirect, url_for, request
import cv2
import tensorflow as tf
import numpy as np
import os


CATEGORIES = ['Dog','Cat']

app = Flask(__name__)

app.secret_key = "qazwsxedcrfvtgbyhnujmik,olbndphaphpanpnn23p"
model = tf.keras.models.load_model('model')

def prepare(path):
    IMG_SIZE = 100
    IMG_ARRAY =  cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(IMG_ARRAY, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)[0]

def empty_static_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                empty_static_folder(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

@app.route('/', methods= ['GET','POST'])
def index():
    empty_static_folder('static')


    if request.method == 'POST':
        image = request.files['file']
        path = 'static/'+str(image.filename)
        image.save(path)
        prediction_input = np.array([prepare(path)])
        prediction = model.predict(prediction_input)
        predicted_class = int(prediction[0][0])
        data = CATEGORIES[predicted_class]
        
        return render_template('image.html', prediction=data, path=path)
    else:
        return render_template('image.html')
    

if __name__ == '__main__':
    app.run(debug=True)  