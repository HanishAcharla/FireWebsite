import os
import ssl
import uuid
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, session
import tensorflow as tf
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Say
from PIL import Image
import logging
import time


# Setup logging
logging.basicConfig(filename='fire_detectio2n.log', level=logging.INFO)
logger = logging.getLogger(__name__)


# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = os.path.join(os.path.dirname(ssl.__file__), 'cert.pem')


app = Flask(__name__)
app.secret_key = "fire_detection_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Configuration - Replace these with environment variables in production
TWILIO_ACCOUNT_SID = 'AC9b6db539daaf8adce9a1b248b8f6b97e'
TWILIO_AUTH_TOKEN = '49c1763c45f51504cc6a20a66828c608'
TWILIO_WHATSAPP_NUMBER = '4155238886'
TWILIO_SMS_NUMBER = '8338979791'
EMERGENCY_NUMBER = '6506276216'


# Model Setup
IMG_SIZE = (224, 224)


# Predefined fire level details with progressive confidence
FIRE_LEVEL_DETAILS = {
    1: {
        "location": "Industrial Zone A",
        "spread_status": "Contained",
        "risk_level": "Low",
        "size": "Small (under 10 sq meters)",
        "smoke_height": "Low (under 5 meters)",
        "stage": "Initial",
        "flammable_materials": "Wood, bushes, trees",
        "confidence": 0.60  # 60% confidence for level 1
    },
    2: {
        "location": "Industrial Zone A",
        "spread_status": "Slowly spreading",
        "risk_level": "Moderate",
        "size": "Medium (10-50 sq meters)",
        "smoke_height": "Medium (5-15 meters)",
        "stage": "Growth",
        "flammable_materials": "Wood, bushes, trees",
        "confidence": 0.75  # 75% confidence for level 2
    },
    3: {
        "location": "Industrial Zone A",
        "spread_status": "Rapidly spreading",
        "risk_level": "High",
        "size": "Large (50-200 sq meters)",
        "smoke_height": "High (15-30 meters)",
        "stage": "Fully developed",
        "flammable_materials": "Wood, bushes, trees",
        "confidence": 0.90  # 90% confidence for level 3
    },
    4: {
        "location": "Industrial Zone A",
        "spread_status": "Uncontrolled",
        "risk_level": "Extreme",
        "size": "Very large (over 200 sq meters)",
        "smoke_height": "Very high (over 30 meters)",
        "stage": "Decay",
        "flammable_materials": "Wood, bushes, trees",
        "confidence": 0.99  # 99% confidence for level 4
    }
}


def load_model():
    try:
        if os.path.exists("fire_model.weights.h5"):
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None
            )
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.load_weights("fire_model.weights.h5")
        else:
            model = tf.keras.Sequential([
                tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
       
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


model = load_model()


def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")
           
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def predict_fire(image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        confidence = 1 - float(prediction[0][0])
        fire_detected = confidence > 0.5
        return confidence, fire_detected
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise


def get_fire_details(level):
    """Return fire details based on the current level"""
    return FIRE_LEVEL_DETAILS.get(level, FIRE_LEVEL_DETAILS[1])


def send_emergency_call(fire_details):
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SMS_NUMBER, EMERGENCY_NUMBER]):
            logger.warning("Twilio credentials not configured")
            return "DEMO_CALL_SID"
           
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = (f"Fire alert detected! "
                  f"Confidence level: {session.get('confidence', 0):.2f}.")
       
        call = client.calls.create(
            to=EMERGENCY_NUMBER,
            from_=TWILIO_SMS_NUMBER,
            twiml=f'<Response><Say>{message}</Say></Response>'
        )
        return call.sid
    except Exception as e:
        logger.error(f"Error sending call: {e}")
        return "ERROR_CALL_SID"
   
def send_detailed_sms(fire_details):
    """Send detailed SMS about fire via Twilio"""
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER, EMERGENCY_NUMBER]):
            print("Twilio credentials not configured. Skipping SMS.")
            return "DEMO_SMS_SID"
           
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
       
        # Craft the SMS message
        message_body = f"FIRE ALERT: {fire_details['location']}\n"
        message_body += f"Spread Status: {fire_details['spread_status']}\n"
        message_body += f"Risk Level: {fire_details['risk_level']}\n"
        message_body += f"Size: {fire_details['size']}\n"
        message_body += f"Smoke Height: {fire_details['smoke_height']}\n"
        message_body += f"Stage: {fire_details['stage']}"
        message_body += f"Flammable Materials: {fire_details['flammable_materials']}"


       
        # Actual Twilio SMS code (commented for testing)
        try:
            message = client.messages.create(
                body=message_body,
                to="whatsapp:+1" + EMERGENCY_NUMBER,
                from_="whatsapp:+1" + TWILIO_WHATSAPP_NUMBER
            )
            return message.sid
        except Exception as twilio_error:
            print(f"Twilio SMS error: {twilio_error}")
            return "ERROR_SMS_SID"
       
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return "ERROR_SMS_SID"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
       
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
       
        try:
            img = Image.open(file)
            img.verify()
            file.seek(0)
        except:
            return jsonify({'error': 'Invalid image file'}), 400
       
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
       
        # Use predefined confidence progression
        confidence = FIRE_LEVEL_DETAILS[1]['confidence']
        fire_detected = confidence > 0.5
       
        session['image_path'] = filename
        session['fire_detected'] = fire_detected
        session['confidence'] = float(confidence)
        session['current_level'] = 1
       
        if fire_detected:
            session['fire_details'] = get_fire_details(1)
            session['call_sid'] = send_emergency_call(session['fire_details'])


            sms_sid = send_detailed_sms(session['fire_details'])
            session['sms_sid'] = sms_sid
       
        return jsonify({
            'success': True,
            'fire_detected': fire_detected,
            'confidence': float(confidence),
            'redirect': '/results'
        })
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Server error processing image'}), 500


@app.route('/next_level')
def next_level():
    try:
        current_level = session.get('current_level', 1)
        next_level = current_level + 1
       
        if next_level > 4:
            return jsonify({'finished': True})
           
        session['current_level'] = next_level
        session['fire_details'] = get_fire_details(next_level)
        session['confidence'] = FIRE_LEVEL_DETAILS[next_level]['confidence']


        send_detailed_sms(session['fire_details'])
       
        session['image_path'] = f"fire level {next_level}.webp"
       
        return jsonify({
            'success': True,
            'image_path': session['image_path'],
            'fire_details': session['fire_details'],
            'confidence': session['confidence'],
            'current_level': next_level,
            'has_next': next_level < 4
        })
    except Exception as e:
        logger.error(f"Next level error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    try:
        return render_template(
            'results.html',
            image_path=session.get('image_path', ''),
            fire_detected=session.get('fire_detected', False),
            confidence=session.get('confidence', 0),
            fire_details=session.get('fire_details', get_fire_details(1)),
            current_level=session.get('current_level', 1)
        )
    except Exception as e:
        logger.error(f"Results error: {e}")
        return render_template('error.html')


if __name__ == '__main__':
    logger.info("Starting Fire Detection Application")
    app.run(debug=True)
