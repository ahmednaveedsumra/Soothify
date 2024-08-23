import cv2
import keras
from keras import utils
from keras.models import load_model
import numpy as np
from keras import models


def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

##model runs for 1000 iterations, gives accuracy. then we pass it the image
##storing it in .h5 file

model = load_model('best_model.h5')
emotionStore = []

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        global T
        ret,frame=self.video.read()
        faces=faceDetect.detectMultiScale(frame, 1.3, 5)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for x,y,w,h in faces:
            x1,y1=x+w, y+h
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 1)
            cv2.line(frame, (x,y), (x+30, y),(255,0,255), 6) #Top Left
            cv2.line(frame, (x,y), (x, y+30),(255,0,255), 6)

            cv2.line(frame, (x1,y), (x1-30, y),(255,0,255), 6) #Top Right
            cv2.line(frame, (x1,y), (x1, y+30),(255,0,255), 6)

            cv2.line(frame, (x,y1), (x+30, y1),(255,0,255), 6) #Bottom Left
            cv2.line(frame, (x,y1), (x, y1-30),(255,0,255), 6)

            cv2.line(frame, (x1,y1), (x1-30, y1),(255,0,255), 6) #Bottom right
            cv2.line(frame, (x1,y1), (x1, y1-30),(255,0,255), 6)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

            ##greying the image
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = keras.utils.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            emotionStore.append(predicted_emotion)
            cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        T = most_frequent(emotionStore)
        print("Most Frequent Emotion is ", T)

        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()

        webbrowser.open((f"https://soundcloud.com/idla/sets/{T}-songs"))



