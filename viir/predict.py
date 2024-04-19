import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Button, filedialog, Label
from tensorflow.keras.models import load_model

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array, ImageTk.PhotoImage(img)

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img, display_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name, display_img

# Load the model and class indices
model = load_model("D:\Project\Plantex-main\Backend\model1.h5")

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

# Function to handle button click event for selecting image
def select_image():
    root = Tk()
    root.withdraw()  # Hide the main window
    image_path = filedialog.askopenfilename(title="Select Image")
    root.destroy()  # Close the hidden window
    if image_path:
        predicted_class_name, display_img = predict_image_class(model, image_path, class_indices)
        image_label.config(image=display_img)
        image_label.image = display_img  # Keep a reference to avoid garbage collection
        prediction_label.config(text="Predicted Class Name: " + predicted_class_name)

# Function to handle button click event for taking picture
def take_picture():
    camera = cv2.VideoCapture(0)
    _, image = camera.read()
    camera.release()
    cv2.destroyAllWindows()
    image_path = 'snapshot.jpg'
    cv2.imwrite(image_path, image)
    predicted_class_name, display_img = predict_image_class(model, image_path, class_indices)
    image_label.config(image=display_img)
    image_label.image = display_img  # Keep a reference to avoid garbage collection
    prediction_label.config(text="Predicted Class Name: " + predicted_class_name)

# Create the GUI
root = Tk()
root.title("Image Classifier")

select_button = Button(root, text="Select Image", command=select_image)
select_button.pack()

take_button = Button(root, text="Take Picture", command=take_picture)
take_button.pack()

# Label to display the image
image_label = Label(root)
image_label.pack()

# Label to display the predicted class name
prediction_label = Label(root, text="")
prediction_label.pack()

root.mainloop()
