import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import losses
import logging
import argparse
import os
import sys

# Configure argument parser
parser = argparse.ArgumentParser(description='Autonomous Car Server')
parser.add_argument('--port', type=int, default=4567, help='Port to run the server on')
parser.add_argument('--model', type=str, default='e:/git_hub/Autonomous-Car/model.h5',
                   help='Path to the model file')
args = parser.parse_args()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def custom_mse(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)
    
import base64
from io import BytesIO
from PIL import Image
import cv2

# Configure Socket.IO with proper namespace
sio = socketio.Server(
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True,
    namespaces=['/telemetry']
)

app = Flask(__name__)
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
    logger.info("Telemetry data received")
    if not data or 'image' not in data:
        logger.error("Invalid telemetry data received")
        return
        
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    logger.info(f"Processed image data, shape: {image.shape}")
    
    try:
        prediction = model.predict(image)
        steering_angle = float(prediction)
        throttle = 1.0 - speed/speed_limit
        logger.info(f'Steering: {steering_angle}, Throttle: {throttle}, Speed: {speed}')
        send_control(steering_angle, throttle)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

@sio.on('connect')
def connect(sid, environ):
    logger.info(f'Client connected: {sid}')
    logger.info(f'Connection details: {environ}')
    send_control(0, 1)

@sio.on('disconnect')
def disconnect(sid):
    logger.info(f'Client disconnected: {sid}')

def send_control(steering_angle, throttle):
    logger.info(f'Sending control: steering={steering_angle}, throttle={throttle}')
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    # Verify model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found at: {args.model}")
        sys.exit(1)

    try:
        logger.info(f"Loading model from {args.model}...")
        model = load_model(args.model, custom_objects={'mse': custom_mse})
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
        
    app = socketio.Middleware(sio, app)
    
    try:
        logger.info(f"Starting server on port {args.port}")
        eventlet.wsgi.server(eventlet.listen(('', args.port)), app)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {args.port} is already in use. Please specify a different port with --port")
        else:
            logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)
