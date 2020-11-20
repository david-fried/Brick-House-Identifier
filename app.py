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
# import streetview
# import itertools
from PIL import UnidentifiedImageError
import shutil
from shutil import copyfile
from datetime import datetime
import subprocess
from my_functions import address_form, image_form
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


# Global variables to track user submissions to the website
IMAGE_SUBMIT_COUNT = 0
ADDRESS_SUBMIT_COUNT = 0
SUBMISSION_TYPE = None # value is None, 'address', or 'image'

app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


# Main page (index.html)
@app.route("/", methods=["GET", "POST"])
def main():

    global SUBMISSION_TYPE
    global ADDRESS_SUBMIT_COUNT
    global IMAGE_SUBMIT_COUNT
    
    # if website user clicks one of the submit buttons
    if request.method == "POST":

        # if user submitted address, call API, save image to:
        # 'static/images/address_submit/{ADDRESS_SUBMIT_COUNT}.jpg'
        if "address" in request.form:

            SUBMISSION_TYPE = 'address'

            submit_address = request.form["address"]

            # return model predictions from user address input
            data, predictions, best_guess_category, address = address_form(model, submit_address, ADDRESS_SUBMIT_COUNT)
            image_upload_path = f'static/images/address_submit/{ADDRESS_SUBMIT_COUNT}.jpg'

        # if user submitted image
        else:

            SUBMISSION_TYPE = 'image'

            request_files = request.files['image']
            image_upload_path = f'static/images/upload_images/{IMAGE_SUBMIT_COUNT}.jpg'
            request_files.save(image_upload_path)

            try:

                image = plt.imread(image_upload_path)

            except UnidentifiedImageError: #submit button was clicked but no image was uploaded

                #need to figure out how to redirect to current location on page
                #https://flask.palletsprojects.com/en/1.1.x/patterns/flashing/
                return redirect('/')

            # return model_predictions from uploaded image
            data, predictions, best_guess_category = image_form(model, image)

        # copy image file to 'new_images' directory after renaming file based on: category of the highest prediction percentage, followed by prediction percentage,
        # and time stamp (year, month, day, hour, second), and if the file was from an address (API call) the address is included at the end
        current_time = f'{datetime.now().year}{datetime.now().month}{datetime.now().day}{datetime.now().hour}{datetime.now().second}'
        prediction = str(int(round(100*max(predictions),0)))
        code = {'Brick': '10', 'Siding': '20', 'Unknown': '00'}
        new_image_name = f'{code[best_guess_category]}_{best_guess_category}_{prediction}_{current_time}'       
        new_image_path = f'new_images/{best_guess_category}/{new_image_name}'
        if SUBMISSION_TYPE == 'address': 
            address = address[0].replace(',', '').replace('.', '').replace(' ', '_')
            new_image_path = new_image_path + '_' + address
        new_image_path_with_extension = new_image_path + '.jpg'
        # copy renamed image file to 'new_images' directory
        copyfile(image_upload_path, new_image_path_with_extension)
                
    else:

        data = {'Heading': '', 'Best_guess': '', 'Brick': '', 'Siding': '', 'Unknown': ''}
   
    return render_template('index.html', data=data)


###########displays image on website##########

#Note: Default image when page loads is "example.jpg" in static/images/upload_images/.
#If user uploads image onto website, uploaded image is renamed f"{IMAGE_SUBMIT_COUNT-1}.jpg" and is saved to, and then retrieved from, static/images/upload_images/ and is displayed on website along with model results.
#If user types address, uploaded image is renamed f"{ADDRESS_SUBMIT_COUNT}.jpg" and is saved to, and then retrieved from, static/images/address_submit/ and is displayed on website along with model results.

@app.route('/load_image')
def load_image():

    global SUBMISSION_TYPE
    global ADDRESS_SUBMIT_COUNT
    global IMAGE_SUBMIT_COUNT

    if SUBMISSION_TYPE == 'address':

        ADDRESS_SUBMIT_COUNT += 1

        return send_from_directory("static/images/address_submit/", f"{ADDRESS_SUBMIT_COUNT-1}.jpg")
    

    elif SUBMISSION_TYPE == 'image':

        IMAGE_SUBMIT_COUNT += 1

        return send_from_directory("static/images/upload_images/", f"{IMAGE_SUBMIT_COUNT-1}.jpg")
    
    else:

        return send_from_directory("static/images/upload_images/", "example.jpg")


# model characteristics website
@app.route('/model_characteristics')
def model_characteristics():
    return render_template('model_characteristics.html')


if __name__ == '__main__':
    app.run(debug=True)