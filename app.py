import os
import pyttsx3
import cv2
import numpy as np
from PIL import ImageOps
import streamlit as st
import tensorflow as ts
from PIL import Image
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

st.set_page_config(
    page_title = "Text Extraction",
    page_icon = ':fish:',
    layout= 'centered'

)

st.title("Text to Speech")


model=keras.models.load_model("characterCNN2.h5")

def result(y):
  if y[0]==0:
    return 'A'
  elif y[0]==1:
    return 'B'
  elif y[0]==2:
    return 'C'
  elif y[0]==3:
    return 'D'
  elif y[0]==4:
    return 'E'
  elif y[0]==5:
    return 'F'
  elif y[0]==6:
    return 'G'
  elif y[0]==7:
    return 'H'
  elif y[0]==8:
    return 'I'
  elif y[0]==9:
    return 'J'
  elif y[0]==10:
    return 'K'
  elif y[0]==11:
    return 'L'
  elif y[0]==12:
    return 'M'
  elif y[0]==13:
    return 'N'
  elif y[0]==14:
    return 'O'
  elif y[0]==15:
    return 'P'
  elif y[0]==16:
    return 'Q'
  elif y[0]==17:
    return 'R'
  elif y[0]==18:
    return 'S'
  elif y[0]==19:
    return 'T'
  elif y[0]==20:
    return 'U'
  elif y[0]==21:
    return 'V'
  elif y[0]==22:
    return 'W'
  elif y[0]==23:
    return 'X'
  elif y[0]==24:
    return 'Y'
  elif y[0]==25:
    return 'Z'

def reverse(str):
    s = ''

    for ch in str:
        s = ch + s
    return s



upload_photo= st.file_uploader("Choose a file")
if upload_photo is not None:
    
    img  = Image.open(upload_photo)
    st.image(img)

    img_array = np.array(img)
    # st.write(type(img_array))
    # st.write(img_array.shape)

    input_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # st.write(input_img.shape)

    def thresholding(image):
        
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, thresh= cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY_INV)
        return thresh


    thresh_img = thresholding(input_img)
    kernel = np.ones((3,85), np.uint8)
    dilated_image = cv2.dilate(thresh_img, kernel, iterations = 1)

    (contours,hierarchy) = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    sorted_conoturs_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1])

    img2 = input_img.copy()
    line_list = []

    for ctr in contours:
        x,y,w,h = cv2.boundingRect(ctr)

        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),3)
        line_list.append([x,y,x+w,y+h])

    if st.button("Translate"):
        img_copy=input_img.copy()

        for line in line_list:
            roi = img_copy[line[1]:line[3], line[0]:line[2]]
            # print(roi)
            thresh_img2 = thresholding(roi)

            kernel = np.ones((3,3), np.uint8)
            dilated_image2 = cv2.dilate(thresh_img2, kernel, iterations = 1)

            (contours,hierarchy) = cv2.findContours(dilated_image2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print(len(contours))

            sorted_conoturs_letters = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[0])


            letter_list = []
            res = ''
            final = ''
            for ctr in contours:
                x,y,w,h = cv2.boundingRect(ctr)

                cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
                letter_list.append([x,y,x+w,y+h])

            for letter in letter_list:
                roi_letter = roi[letter[1]:letter[3], letter[0]:letter[2]]
                # print(len(roi_letter))
                cv2.imwrite('./letter.png', roi_letter)

                image_receive = tf.keras.preprocessing.image.load_img("./letter.png", target_size=(20,20))
                x = tf.keras.preprocessing.image.img_to_array(image_receive)
                x = np.expand_dims(x,axis=0)
                x /= 255.0
                images = np.vstack([x])# [1 2 3 4 5 6]
                classes = model.predict(x)
                y_classes=classes.argmax(axis=-1)
            
                res += result(y_classes)
            rev_result = reverse(res)
            st.write(rev_result)

            engine = pyttsx3.init()
            # print(engine)
            # answer = input("Eneter something : ")
            engine.setProperty('rate',150)
            engine.say(rev_result)
            engine.runAndWait()
        
