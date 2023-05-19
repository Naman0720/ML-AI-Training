import cv2
import matplotlib.pyplot  as plt
fd= cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  #github file
)
sd=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

vid=cv2.VideoCapture(0)
captured=False
vid.isOpened()
flag,img=vid.read()
from time import sleep

while not captured:
    flag,img=vid.read()
    if flag:
        seq=0
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #to convert in gray

        th,img_binary=cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY) #to convert in binary

        x1,y1,w,h = (200,400,300,400)
        # smile=sd.detectMultiScale(img_gray,1.1,5)
        # for x1,y1,w,h in smile:
        #     img_cropped = img[y1:y1+h, x1:x1+w , :] #crops the image

        #     cv2.rectangle(
        #                 img, 
        #                 pt1=[x1,y1], pt2= [x1+w, y1+h], 
        #                 color=(0,0,255),
        #                 thickness=10)
        
        faces= fd.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50,50)) #face detection  
                                                   #scaleFactor=1.1{search in face}
                                                   #minNeighbors=5{need to process the face }
                                                   #minSize=size of widhth & height{no of pixels}
         

        for x1,y1,w,h in faces:
            face = img[y1:y1+h, x1:x1+w , :].copy() #crops the image
            smiles = sd.detectMultiScale(face,1.1,15,minSize=(50,50))
            print(len(smiles))

            if len(smiles)==1:
                seq+=1
                if seq==2:
                    captured=cv2.imwrite('smile.png',img)
                    break
                

                #xs,ys,ws,hs = (200,400,300,4)
            xs,ys,ws,hs = smiles[0]
            cv2.rectangle(
                            img, 
                            pt1=(xs+x1,ys+ys), pt2= (xs+x1+ws, y1+ys+hs), 
                            color=(0,255,0),
                            thickness=10)
            cv2.rectangle(
                        img, 
                        pt1=(x1,y1), pt2= (x1+w, y1+h), 
                        color=(0,0,255),
                        thickness=10
                        )  #higlights the particular part
            #cv2.imwrite('smile.png',img)

            cv2.imshow('Preview',img)
        key=cv2.waitKey(1)
        if key==ord(' '): #will close the window when pressed spacebar
            break
    else :
        break
    sleep(0.1) #for delay
vid.release() #turns off the camera
cv2.destroyAllWindows()
#cv2.waitKey(1)    # only for Mac Os ,not for windows.
