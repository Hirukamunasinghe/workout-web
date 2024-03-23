from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import io
import base64

app = Flask(__name__)

# Load your trained model
model_path = './modal/Image_classify.keras'  # Update this path
model = load_model(model_path)

# Directly using the labels as a list
data_cat = ['zArmCurlMuchine', 'zHipAbduction', 'zLegExtension', 'zLyingLegCurl', 'zback_extension', 'zcable-machine', 'zcd', 'zchest_press', 'zdown', 'zpd', 'zpower_leg_press', 'zsmith_machine', 'zsp']

img_height = 180
img_width = 180

@app.route('/', methods=['GET'])
def index():
    # Render the upload form
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        # Convert the FileStorage object to a BytesIO object
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)

        # Load and prepare the image
        image_load = tf.keras.utils.load_img(
            in_memory_file, target_size=(img_height, img_width), color_mode='rgb'
        )
        img_array = tf.keras.utils.img_to_array(image_load)
        img_batch = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_batch)
        score = tf.nn.softmax(predictions[0])
        result = data_cat[np.argmax(score)]
        accuracy = round(100 * np.max(score), 2)

        # Prepare image data for displaying
        in_memory_file.seek(0)
        image_data = base64.b64encode(in_memory_file.read()).decode('utf-8')

        return render_template('index.html', result=result, accuracy=accuracy, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)