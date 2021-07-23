import cv2
import face_recognition

image=face_recognition.load_image_file("robert.jpg")
image_encodings=face_recognition.face_encodings(image)[0]

image_test=face_recognition.load_image_file("robert.png")
image_test_encodings=face_recognition.face_encodings(image_test)[0]

results=face_recognition.compare_faces([image_encodings],image_test_encodings)
print(results)
cv2.imshow("robert",image)
cv2.waitKey(0)
