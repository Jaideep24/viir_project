import tensorflow as tf
model = tf.keras.models.load_model(r"C:\Users\Inder Pal Singh\Desktop\Jaideep\viir_project\viir\model1.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model) 
tfmodel = converter.convert() 
open ('model.tflite' , "wb") .write(tfmodel)