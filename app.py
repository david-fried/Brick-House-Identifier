from flask import Flask, render_template, \
    request, send_from_directory, redirect, url_for
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import urllib.request
import os
import google_streetview.api
import glob
from PIL import UnidentifiedImageError
import shutil
from shutil import copyfile
from datetime import datetime
import subprocess
from my_functions import model_classifications, address_form, create_full_image_path
from config import gkey

# Image processing model
model = Sequential()
model.add(Conv2D(filters=2, kernel_size=2, padding='same',
                 activation='relu', input_shape=(500, 500, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=4, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=12, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
latest_model = glob.glob('static/model/*.hdf5')[0]
print(f'\nUsing the following image classification model:\n\n\t{latest_model}.\n')
model.load_weights(latest_model)


app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


# Main page (index.html)
@app.route("/", methods=["GET", "POST"])
def main():
    
    # if website user clicks one of the submit buttons
    if request.method == "POST":

        # if user submitted address, call API, save image to:
        # 'static/images/address_submit/{ADDRESS_SUBMIT_COUNT}.jpg'
        if "address" in request.form:

            SUBMISSION_TYPE = 'address'

            user_typed_address = request.form["address"]

            # return model predictions from user address input
            data, predictions, best_guess_category, address, URL = address_form(model, user_typed_address)

            image_path = URL

        # if user submitted image
        else:

            SUBMISSION_TYPE = 'image'

            request_files = request.files['image']

            temp = 'static/uploaded_images/temp.jpg'

            request_files.save(temp)

            try:

                image = plt.imread(temp)

            except UnidentifiedImageError: #submit button was clicked but no image was uploaded

                #need to figure out how to redirect to current location on page
                #https://flask.palletsprojects.com/en/1.1.x/patterns/flashing/
                return redirect('/')

            # return model_predictions from uploaded image
            data, predictions, best_guess_category = model_classifications(model, image)

        # naming file based on: category of the highest prediction percentage, followed by prediction percentage,
        # and time stamp (year, month, day, hour, second), and if the file was from an address (API call) the address is included at the end
        
        if SUBMISSION_TYPE == 'image':
        
            image_path = create_full_image_path(predictions, best_guess_category, address = None)

            copyfile(temp, image_path)
        
        else:

            saved_image_path = create_full_image_path(predictions, best_guess_category, address)

            urllib.request.urlretrieve(URL, saved_image_path)

        form_submit = "form_anchor"
                
    else:

        form_submit = None

        data = {'Heading': '', 'Best_guess': '', 'Brick': '', 'Siding': '', 'Unknown': ''}

        image_path = 'static/images/uploaded_images/example.jpg'

    return render_template('index.html', data=data, form_submit = form_submit, image_path=image_path)


# model characteristics website
@app.route('/model_characteristics')
def model_characteristics():
    return render_template('model_characteristics.html')


if __name__ == '__main__':
    app.run(debug=True)