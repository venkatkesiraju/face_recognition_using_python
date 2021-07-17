def video(i):

    import cv2
    import numpy as np
    p=["b1.jpg","b2.jpg","b3.jpg","b4.jpg",'b5.jpg',"b6.jpg"]
    image=cv2.imread(p[i-1])
    cap=cv2.VideoCapture(0)
    
    while True:
        flag,frame=cap.read()
        if not flag:
            print("could not acess camera")
            break
        image=cv2.resize(image,(frame.shape[1],frame.shape[0]))
        #cv2.imshow("resized image",image)
        blended_image=cv2.addWeighted(frame,0.8,image,0.2,gamma=0.1)
        cv2.imshow("frame",blended_image)
        cv2.waitKey(10)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()   

while True:
    i=int(input("enter background image number you want"))
    video(i)



