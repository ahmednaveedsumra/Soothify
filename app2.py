import sys
import webbrowser
from flask import Flask, render_template, Response, request,flash
from camera import Video
import cv2
import keras
from keras import utils
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import arduinoTest
from keras.preprocessing import image
from time import sleep
import os
import firebase_admin
from firebase_admin import credentials, db, auth
from datetime import datetime
current_directory = os.path.dirname(os.path.realpath(__file__))
key_file_path = os.path.join(current_directory, 'fyp2024-65275-firebase-adminsdk-h2hz7-dc1b2ec062.json')

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("model.h5")
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            emotionStore.append(label)
            emotion = most_frequent(emotionStore)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()

app = Flask(__name__, static_url_path='')
app.secret_key = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

cred = credentials.Certificate(key_file_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fyp2024-65275-default-rtdb.firebaseio.com/'
})

def gen(camera):
    x=0
    while x<100:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
            b'Content-Type:  image/jpeg\r\n\r\n' + frame +
            b'\r\n\r\n')
        x=x+1

    if emotion == 'Neutral':
        neutral = ["https://soundcloud.com/soothingrelaxation/sets/relaxing-music-mixes",
                   "https://soundcloud.com/soothingrelaxation/sets/long-relaxing-music-extended",
                   "https://soundcloud.com/omasan-o/sets/chill-relaxing-songs"]

        import random
        song = random.choice(neutral)
        webbrowser.open(song)
        arduinoTest.func1()
    elif emotion == 'Happy':
        happy = ["https://soundcloud.com/idla/sets/happy-songs",
                 "https://soundcloud.com/alexrainbirdmusic/sets/good-vibes-a-happy-indie-pop",
                 "https://soundcloud.com/effyfrenchdreamer/sets/happy-songs"]

        import random
        song = random.choice(happy)
        webbrowser.open(song)
        arduinoTest.func3()
    elif emotion == 'Angry':
        webbrowser.open(f"https://soundcloud.com/amna-anie-684178201/sets/jolly-songs")
        arduinoTest.func3()
    elif emotion == 'sad':
        webbrowser.open(f"https://soundcloud.com/ishu-kumar-307962012/sets/happy-songs-2022-good-vibes")
        arduinoTest.func3()

    elif emotion == "surprise":
        webbrowser.open(f"https://soundcloud.com/luye_exclamation_mark/element-of-suprise")
        arduinoTest.func2()
    elif emotion == "fear":
        webbrowser.open(f"https://soundcloud.com/fearlessmotivation/sets/epic-motivational-music")
    elif emotion == "disgust":
        webbrowser.open(f"https://soundcloud.com/amna-anie-684178201/sets/jolly-songs")
    else:
        print("Sorry dear")
    sys.exit(app)

def check_login(email, password):
    try:
        user = auth.get_user_by_email(email)
        # User exists, now check if the passwords match
        if user.email == email:
            # User exists and email matches, now check password
            if auth.verify_password(password, user.password):
                flash("Login successful.", "success")  # "success" is the category of the flash message
                return render_template('signup_page.html')
            else:
                print("Password incorrect.")
        else:
            flash("Password incorrect.", "error")  # "error" is the category of the flash message
            print("User not found.")
    except auth.UserNotFoundError:
        print("User not found.")
def save_signup_data(name, email, password):
    current_datetime = datetime.now()
    created_at = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    ref = db.reference('users')  # Reference to the 'users' table in the database
    ref.push({
        'name': name,
        'email': email,
        'password': password,
        'createdAt': created_at
    })

@app.route('/', methods=['GET'])
def signup_page():
    print("get login")
    return render_template('signup_page.html')

@app.route('/', methods=['POST'])
def login_firebase():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_ref = db.reference('users').order_by_child('email').equal_to(email).get()
        if user_ref:
            user_id, user_data = next(iter(user_ref.items()))
            if user_data.get('password') == password:
                return render_template('index.html')
            else:
                flash("Invalid email or password", "error")
                return render_template('signup_page.html')
        else:
            flash("Invalid email or password", "error")
            return render_template('signup_page.html')

@app.route('/signup', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def process_signup_form():
    if request.method == 'POST':
        print("Name:", request.form)
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        save_signup_data(name, email, password)
        flash("Signup data saved successfully", "success")
        return render_template('login.html')
    else:
        return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/player')
def player():
    return render_template('player.html')

@app.route('/video')
def video():
    return Response(gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)
sys.exit()
