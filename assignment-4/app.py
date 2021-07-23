import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import face_recognition


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation

model_face_mesh = mp_face_mesh.FaceMesh()
model_face_detection=mp_face_detection.FaceDetection()

st.title("OpenCV Operations")
st.subheader("Image operations")


st.write("This application performs various operations with OpenCV")


add_selectbox = st.sidebar.selectbox(
    "What operations you would like to perform?",
    ("About", "Face Detection", "Face Recognition", "Selfie Segmentation")
)
image_file_path = st.sidebar.file_uploader("Upload image")
if add_selectbox == "About":
    st.write("This application is used for face recognition,face detection and selfie segmentation using streamlit.")

elif add_selectbox =="Face Recognition":
    image_file_path_1 = st.sidebar.file_uploader("Upload image which you want to detect faces",key="face")
    if image_file_path_1 is not None:
        image_1= np.array(Image.open(image_file_path_1))
        st.sidebar.image(image_1)
        image_2=np.array(Image.open(image_file_path))
        image_1_encodings=face_recognition.face_encodings(image_1)[0]
        image_2_encodings=face_recognition.face_encodings(image_2)[0]
        results=face_recognition.compare_faces([image_1_encodings],image_2_encodings)
        #print(results)
        if results[0]==True:
            st.image(image_1)
            st.image(image_2)
            st.write("both faces are same")
        else:
            st.image(image_1)
            st.image(image_2)
            st.write("both faces are not same")

elif add_selectbox == "Face Detection":
    #image_file_path = st.sidebar.file_uploader("Upload image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model_face_detection.process(image)

        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
        st.image(image)
elif add_selectbox =="Selfie Segmentation":
    background_image = st.sidebar.radio("Choose background image",
                                 ("1","2")
                                 )
    color_schemes = st.sidebar.radio("Choose your color",
                                 ("B", "G", "R")
                                 )

    if color_schemes == "B":
       #image_file_path = st.sidebar.file_uploader("Upload image")
       if image_file_path is not None:
          image = np.array(Image.open(image_file_path))
          st.sidebar.image(image)
          zeros = np.zeros((image.shape[0], image.shape[1]), np.uint8)
          b,g,r = cv2.split(image)
          blue = cv2.merge([zeros, zeros, b])
          st.image(blue)

    elif color_schemes == "G":
        #image_file_path = st.sidebar.file_uploader("Upload image")
        if image_file_path is not None:
            image = np.array(Image.open(image_file_path))
            st.sidebar.image(image)
            zeros = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            b,g,r = cv2.split(image)
            green = cv2.merge([zeros, g, zeros])
            st.image(green)
    elif color_schemes == "R":
        #image_file_path = st.sidebar.file_uploader("Upload image")
        if image_file_path is not None:
            image = np.array(Image.open(image_file_path))
            st.sidebar.image(image)
            zeros = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            b,g,r = cv2.split(image)
            red = cv2.merge([r, zeros, zeros])
            st.image(red)
    with mp_selfie_segmentation.SelfieSegmentation(
      model_selection=0) as selfie_segmentation:
      #image_file_path = st.sidebar.file_uploader("Upload image")
      if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        BG_COLOR1 = (192, 192, 192) # gray
        BG_COLOR2 = (255, 0, 0) #red
        MASK_COLOR = (255, 255, 255) # white


        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        if background_image=="1":
            BG_COLOR=BG_COLOR1
        elif background_image=="2":
            BG_COLOR=BG_COLOR2
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
        st.image( output_image)
