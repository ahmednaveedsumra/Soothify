import sys
import webbrowser
from flask import Flask, render_template, Response
from camera import Video
import cv2
import keras
from keras import utils
from keras.models import load_model
import numpy as np
import arduinoTest



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
model = load_model("best_model.h5")
emotionStore = []
class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        global emotion
        global i
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
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = keras.utils.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 250
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'neutral', 'happy', 'sad', 'neutral', 'fear')
            predicted_emotion = emotions[max_index] ##use this to write
            emotionStore.append(predicted_emotion)
            emotion = most_frequent(emotionStore)
            cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()

app = Flask (__name__,
        static_url_path='')
def gen(camera):
    x=0
    while x<50:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')
        x=x+1
    print(emotion)
    if emotion == 'neutral':
        neutral = ["https://soundcloud.com/soothingrelaxation/sets/relaxing-music-mixes",
                   "https://soundcloud.com/soothingrelaxation/sets/long-relaxing-music-extended",
                   "https://soundcloud.com/omasan-o/sets/chill-relaxing-songs"]

        import random
        song = random.choice(neutral)
        webbrowser.open(song)
        arduinoTest.func1()
    elif emotion == 'happy':
        happy = ["https://soundcloud.com/idla/sets/happy-songs",
                 "https://soundcloud.com/alexrainbirdmusic/sets/good-vibes-a-happy-indie-pop",
                 "https://soundcloud.com/effyfrenchdreamer/sets/happy-songs"]

        import random
        song = random.choice(happy)
        webbrowser.open(song)
        arduinoTest.func3()
    elif emotion == 'angry':
        webbrowser.open(f"https://soundcloud.com/amna-anie-684178201/sets/jolly-songs")
        arduinoTest.func3()
    elif emotion == 'sad':
        webbrowser.open(f"")
    elif emotion == "surprise":
        webbrowser.open(f"")
        arduinoTest.func2()
    elif emotion == "fear":
        webbrowser.open(f"")
    elif emotion == "disgust":
        webbrowser.open(f"")
    else:
        print("Sorry")
    sys.exit(app)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/player')
def player():
    return render_template('player.html')
@app.route('/video')
def video():
    return Response(gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug= True)
sys.exit()






