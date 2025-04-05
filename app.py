import os
import ssl
import uuid
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, session
import tensorflow as tf
from twilio.rest import Client
import requests
from twilio.twiml.voice_response import VoiceResponse, Say
import base64
# test commit


# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = os.path.join(os.path.dirname(ssl.__file__), 'cert.pem')

app = Flask(__name__)
app.secret_key = "fire_detection_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Twilio configuration
TWILIO_ACCOUNT_SID = 'AC9b6db539daaf8adce9a1b248b8f6b97e'  # Add your Twilio account SID
TWILIO_AUTH_TOKEN = '49c1763c45f51504cc6a20a66828c608'   # Add your Twilio auth token
TWILIO_WHATSAPP_NUMBER = '4155238886' # Add your Twilio WhatsApp phone number
TWILIO_SMS_NUMBER = '8338979791' # Add your Twilio phone number
EMERGENCY_NUMBER = '6506276216' # Add emergency contact number

# Palantir AIP configuration (placeholder)
PALANTIR_API_KEY = ''    # Add your Palantir API key
PALANTIR_API_URL = 'https://api.palantir.com/analysis'  # Replace with actual API endpoint

# Model Setup
IMG_SIZE = (224, 224)

def load_model():
    """Load the fire detection model with error handling"""
    try:
        # Option 1: Load a saved model if available
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
            print("Loaded saved model weights.")
        else:
            # Option 2: Use a local copy of MobileNetV2 weights if downloading fails
            print("No saved model found. Creating new model with pretrained weights.")
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
            # Since we don't have trained weights, this mock model will randomly predict for demo purposes
            print("Warning: Using untrained model for demonstration purposes.")
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create an emergency fallback model that always predicts fire (for demo purposes)
        print("Creating fallback model.")
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

# Initialize the model
model = load_model()

def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
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
        print(f"Error preprocessing image: {e}")
        # Return a blank image if processing fails
        return np.zeros((1, 224, 224, 3))

def predict_fire(image_path):
    """Predict if there's a fire in the image"""
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        confidence = 1 - float(prediction[0][0])
        # Lower values (close to 0) indicate fire detected, higher values (close to 1) indicate no fire
        fire_detected = confidence > 0.5
        return confidence, fire_detected
    except Exception as e:
        print(f"Error making prediction: {e}")
        # For demo purposes, assume fire in case of error
        return 0.3, True

def analyze_fire_with_palantir(image_path):
    """Send image to Palantir AIP for detailed fire analysis"""
    # This is a placeholder implementation
    # In a real implementation, you would send the image to Palantir API
    
    try:
        # For demonstration, we'll use a mock response
        mock_response = {
            "location": "I-5 Junction with Highway 405",
            "spread_status": "Early",
            "risk_level": "High",
            "size": "Approximately 50 square meters",
            "smoke_height": "20 meters",
            "stage": "Initial ignition",
            "flammable_materials": "Heavily flammable area"
        }
        
        # In a real implementation:
        # with open(image_path, 'rb') as img_file:
        #     img_data = base64.b64encode(img_file.read()).decode('utf-8')
        # 
        # headers = {
        #     'Authorization': f'Bearer {PALANTIR_API_KEY}',
        #     'Content-Type': 'application/json'
        # }
        # 
        # payload = {
        #     'image': img_data,
        #     'analysis_type': 'fire_analysis'
        # }
        # 
        # response = requests.post(PALANTIR_API_URL, headers=headers, json=payload)
        # return response.json()
        
        return mock_response
    except Exception as e:
        print(f"Error analyzing with Palantir: {e}")
        # Return default values if analysis fails
        return {
            "location": "Unknown Location",
            "spread_status": "Unknown",
            "risk_level": "Unknown",
            "size": "Unknown",
            "smoke_height": "Unknown",
            "stage": "Unknown",
            "flammable_materials": "Unknown"
        }

def send_emergency_call(fire_details):
    """Send emergency call via Twilio with fire details"""
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SMS_NUMBER, EMERGENCY_NUMBER]):
            print("Twilio credentials not configured. Skipping emergency call.")
            return "DEMO_CALL_SID"
            
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create TwiML response for the call
        response = VoiceResponse()
        
        # Craft the message
        message = f"Fire detected at the {fire_details['location']}. "
        message += f"Spread status is {fire_details['spread_status']}. "
        message += "Immediate response is needed. "
        message += f"{fire_details['flammable_materials']}."

        print(message)
        
        # Actual Twilio call code (commented out for demo)
        call = client.calls.create(
            to=EMERGENCY_NUMBER,
            from_=TWILIO_SMS_NUMBER,
            twiml=f'<Response><Say>{message}</Say></Response>'
        )
        return call.sid

    except Exception as e:
        print(f"Error sending emergency call: {e}")
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
        message_body += f"Status: {fire_details['spread_status']}\n"
        message_body += f"Risk: {fire_details['risk_level']}\n"
        message_body += f"Size: {fire_details['size']}\n"
        message_body += f"Smoke Height: {fire_details['smoke_height']}\n"
        message_body += f"Stage: {fire_details['stage']}\n"
        message_body += f"Area: {fire_details['flammable_materials']}"
        
        # Actual Twilio SMS code (commented out for demo)
        message = client.messages.create(
            body=message_body,
            to="whatsapp:+1" + EMERGENCY_NUMBER,
            from_="whatsapp:+1" + TWILIO_WHATSAPP_NUMBER
        )
        return message.sid
        
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
        
        # Save the uploaded file
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze image with your model
        confidence, fire_detected = predict_fire(filepath)
        
        # Store results in session
        session['image_path'] = filepath
        session['fire_detected'] = fire_detected
        session['confidence'] = float(confidence)
        
        if fire_detected:
            # Get detailed analysis from Palantir AIP
            fire_details = analyze_fire_with_palantir(filepath)
            session['fire_details'] = fire_details
            
            # Send emergency call
            call_sid = send_emergency_call(fire_details)
            session['call_sid'] = call_sid
            
            # Send detailed SMS
            sms_sid = send_detailed_sms(fire_details)
            session['sms_sid'] = sms_sid
        
        # Redirect to results page
        return jsonify({
            'success': True,
            'fire_detected': fire_detected,
            'confidence': float(confidence),
            'redirect': '/results'
        })
    except Exception as e:
        print(f"Error processing upload: {e}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

@app.route('/results')
def results():
    try:
        # Get data from session
        image_path = session.get('image_path', '')
        fire_detected = session.get('fire_detected', False)
        confidence = session.get('confidence', 0)
        
        relative_image_path = image_path.replace('static/', '')
        
        # If fire detected, get the fire details
        fire_details = session.get('fire_details', {}) if fire_detected else {}
        
        return render_template(
            'results.html',
            image_path=relative_image_path,
            fire_detected=fire_detected,
            confidence=confidence,
            fire_details=fire_details
        )
    except Exception as e:
        print(f"Error rendering results: {e}")
        return "An error occurred. Please try again."

if __name__ == '__main__':
    print("Starting Fire Detection Web Application...")
    app.run(debug=True)