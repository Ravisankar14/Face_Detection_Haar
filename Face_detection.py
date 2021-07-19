import cv2

print("Loading the Face Detector")

#Loding face detection xml
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Selecting the video source
cap=cv2.VideoCapture(0)

#getting height and width of the video frame
height=int(cap.get(3))
width=int(cap.get(4))

#setting the fourcc format
fourcc=cv2.VideoWriter_fourcc(*'MP4V')

#Creating the object for VideoWriter
out=cv2.VideoWriter("output.mp4",fourcc,20.0,(height,width))

#Creating the empty window
cv2.namedWindow("Image",cv2.WINDOW_NORMAL)

#Creating loop for processing frames
while(True):
        #Reading one frame
        ret,frame=cap.read()

        #Converting the RGB frame into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Detecting the faces
        rects = detector.detectMultiScale(gray, scaleFactor=1.3,
                minNeighbors=4, minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE)

        #Printing number of faces detected
        print("{} faces detected".format(len(rects)))

        #Saving number of faces
        num_faces=len(rects)

        #Checking face detected or not
        if num_faces > 0:
                frame = cv2.putText(frame, 'Face Detected',(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(0,255,0),2,cv2.LINE_AA)
        else:
                frame = cv2.putText(frame, 'Face not Detected',(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(0,0,255),2,cv2.LINE_AA)
                
        #Creating loop for prcessing each face
        for (x, y, w, h) in rects:
                #Drawing the rectangle for the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)

        #Writing the frames into video
        out.write(frame)
        
        #Showing the output in the window       
        cv2.imshow("Image",frame)

        #Creating waitKey for keyboard response and Quiting the window 
        if cv2.waitKey(1) & 0XFF==ord("q"):
                #Breaking the while loop after pressing the q
                break

#Releasing the capture object
cap.release()

#Releasing the writer object
out.release()

#Destroying all the windows
cv2.destroyAllWindows()
































