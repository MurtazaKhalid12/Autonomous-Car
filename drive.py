import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import losses

def custom_mse(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server(cors_allowed_origins='*', logger=True, engineio_logger=True)

app = Flask(__name__) #'__main__'
speed_limit = 10
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    print("Received image data, shape:", image.shape)
    prediction = model.predict(image)
    print("Model prediction:", prediction)
    steering_angle = float(prediction)
    throttle = 1.0 - speed/speed_limit
    print('Steering: {}, Throttle: {}, Speed: {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)



@sio.on('connect')
def connect(sid, environ):
    print('Client connected:', sid)
    print('Connection details:', environ)
    send_control(0, 1)

@sio.on('disconnect')
def disconnect(sid):
    print('Client disconnected:', sid)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('e:/git_hub/Autonomous-Car/model.h5', custom_objects={'mse': custom_mse})
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)