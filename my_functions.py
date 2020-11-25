from flask import Flask, render_template, request, send_from_directory, redirect
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import urllib.request
import requests
import google_streetview.api
import glob
from config import gkey
from datetime import datetime
from urllib.error import HTTPError

# Statuses from Google Streetview Metadata
# https://developers.google.com/maps/documentation/streetview/metadata
"""
"OK"	Indicates that no errors occurred; a panorama is found and metadata is returned.
"ZERO_RESULTS"	Indicates that no panorama could be found near the provided location. This may occur if a non-existent or invalid panorama ID is given.
"NOT_FOUND"	Indicates that the address string provided in the location parameter could not be found. This may occur if a non-existent address is given.
"OVER_QUERY_LIMIT"	Indicates that you have exceeded your daily quota or per-second quota for this API.
"REQUEST_DENIED"	Indicates that your request was denied. This may occur if you did not use an API key or client ID, or if the Street View Static API is not activated in the Google Cloud Platform Console project containing your API key.
"INVALID_REQUEST"	Generally indicates that the query parameters (address or latlng or components) are missing.
"UNKNOWN_ERROR"	Indicates that the request could not be processed due to a server error. This is often a temporary status. The request may succeed if you try again.
"""

#returns classifications for an (uploaded) image

def model_classifications(model, image):
    
    resized_image = resize(image, (400, 400, 3))
    
    data = {}
    
    predictions = model.predict(np.array([resized_image]))[0]
    
    predictions = list(predictions)
    
    best_guess_value = max(predictions)
    
    best_guess_index = predictions.index(best_guess_value)
    
    classifications = {0: 'Brick', 1: 'Siding', 2: 'Unknown'}
    
    best_guess_category = classifications[best_guess_index]

    data['Heading'] = 'THE CLASSIFICATION OF THIS PROPERTY IS'
    
    if best_guess_value >= 0.395:
    
        data['Best_guess'] = best_guess_category + '.'
   
    else:
    
        data['Best_guess'] = 'Unknown.'

    for i, prediction in enumerate(predictions):

        # print(classifications[i], predictions[i])
     
        data[classifications[i]] = f'{classifications[i]}: {int(round(100*prediction,0))}%'
 
    return data, predictions, best_guess_category


# Using user-typed address, retrieves image URL of house from GoolgeStreetView API call
# and classifies image as brick, siding, or unknown.

def address_form(model, user_typed_address):

    #this is the first part of the streetview, url up to the address, this url will return a 600x600px image
    pre='https://maps.googleapis.com/maps/api/streetview'

    img_rqst = '?size=600x600&location='

    address = user_typed_address.replace(' ', '+')
    
    suf=f"&key={gkey}&fov=60"

    URL = pre + img_rqst + address + suf

    request = pre + '/metadata' + img_rqst + address + suf

    status = requests.get(request).json()['status']

    image = plt.imread(urllib.request.urlopen(URL), format='JPG')

    data, predictions, best_guess_category = model_classifications(model, image)

    if status != 'OK':

        data = {'Heading': '', 'Best_guess': '', 'Brick': '', 'Siding': '', 'Unknown': ''}

        data['Best_guess'] = 'Please enter a valid address or location.'

    return (data, predictions, best_guess_category, address, URL)


#creates path for where uploaded images will be saved

def create_full_image_path(predictions, best_guess_category, address):
    
    current_time = f'{datetime.now().year}{datetime.now().month}{datetime.now().day}{datetime.now().hour}{datetime.now().second}'
    
    prediction = str(int(round(100*max(predictions),0)))
    
    code = {'Brick': '10', 'Siding': '20', 'Unknown': '00'}
    
    image_file_name = f'{code[best_guess_category]}_{best_guess_category}_{prediction}_{current_time}'       
    
    image_file_path = f'static/uploaded_images/{best_guess_category}/{image_file_name}'
    
    if address is not None:
     
        address = address[0].replace(',', '').replace('.', '').replace(' ', '_')
     
        image_file_path = image_file_path + '_' + address

    return image_file_path + '.jpg'