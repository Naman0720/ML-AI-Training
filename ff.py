import cv2
vid = cv2.VideoCapture(0)
fd = cv2.CascadeClassifier(
    cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
frameCount =0
name = input('enter your name:')
while True:
    flag, img=vid.read()
    if flag:
        faces= fd.detectMultiScale(
            cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50,50)
        )
        if len(faces)==1:
            x,y,w,h =faces(0)
            img_face = img[y:y+h, x:x+w, :].copy()
            cv2.imwrite('',img_face)
            frameCount+=1
        cv2.imshow('preview', img)
        key = cv2.waitKey(1)
        if key==ord('q'):
            break
cv2.destroyAllWindows()