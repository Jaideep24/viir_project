from ast import For
from dataclasses import fields
from re import template
from urllib import request
from django.shortcuts import render
import requests
from .forms import *
from .models import *
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import CreateView
from django.views.generic.list import ListView
import os
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Button, filedialog, Label
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
# Create your views here.
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

interpreter = tf.lite.Interpreter(model_path=r"viir\viir.tflite")
interpreter.allocate_tensors()

# Function to predict the class of an image
def predict_image_class(image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class_index = np.argmax(output_data, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name
class_indices = {0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}
@csrf_exempt
def home(request):
    result=""
    if request.method == "POST":
        print(request.FILES)
        print(request.FILES['img'])
        form = Imageaform(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            address="media/"+request.FILES['img'].name
            print(address)
            result=predict_image_class(address,class_indices)
            return JsonResponse({'result': result})
    form = Imageaform()
    return (render(request,"viir/home.html",{"form":form,"result":result}))