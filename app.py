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
import openai
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

# OpenAI configuration
OPENAI_API_KEY = "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"
# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

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

def analyze_fire_with_openai(image_path):
    """Send image to OpenAI for detailed fire analysis"""
    try:
        # Check if image path exists
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            # Use mock response since we can't access the image
            return {
                "location": "Interstate-5 junction with Highway 42",
                "terrain": "Hilly with dry vegetation",
                "size": "Approximately 2-3 acres",
                "fuel_type": "Dry brush and small trees",
                "smoke_height": "30-40 meters",
                "stage": "Initial growth phase"
            }
        
        if not OPENAI_API_KEY:
            print("OpenAI API key not configured. Using mock response.")
            # Hardcoded location as requested
            return {
                "location": "Interstate-5 junction with Highway 42",
                "terrain": "Hilly with dry vegetation",
                "size": "Approximately 2-3 acres",
                "fuel_type": "Dry brush and small trees", 
                "smoke_height": "30-40 meters",
                "stage": "Initial growth phase"
            }
        
        try:
            # Prepare image for OpenAI API (base64 encode)
            with open(image_path, 'rb') as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Call OpenAI API with vision capability
            response = openai.chat.completions.create(
                model="gpt-4-vision-preview",  # Choose the appropriate model with vision capabilities
                messages=[
                    {
                        "role": "system",
                        "content": """You are a fire analysis expert. Analyze the image provided and extract the following information:
                        1. Terrain type
                        2. Fire size estimate
                        3. Fuel type (what is burning)
                        4. Smoke height estimate
                        5. Fire stage (initial, growing, fully developed, etc.)
                        
                        Return your analysis in JSON format with the following keys:
                        terrain, size, fuel_type, smoke_height, stage"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this fire image and provide details as specified."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Parse the response to get JSON
            content = response.choices[0].message.content
            try:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Attempt to parse the entire content as JSON
                    analysis = json.loads(content)
            except json.JSONDecodeError:
                print("Failed to parse JSON from OpenAI response. Using structured extraction.")
                # Extract information using regex patterns
                analysis = {
                    "terrain": extract_field(content, "terrain", "Hilly with dry vegetation"),
                    "size": extract_field(content, "size", "Approximately 2-3 acres"),
                    "fuel_type": extract_field(content, "fuel_type", "Dry brush and small trees"),
                    "smoke_height": extract_field(content, "smoke_height", "30-40 meters"),
                    "stage": extract_field(content, "stage", "Initial growth phase")
                }
            
        except Exception as api_error:
            print(f"OpenAI API error: {api_error}")
            # Use mock data if API call fails
            analysis = {
                "terrain": "Hilly with dry vegetation",
                "size": "Approximately 2-3 acres",
                "fuel_type": "Dry brush and small trees",
                "smoke_height": "30-40 meters", 
                "stage": "Initial growth phase"
            }
        
        # Add hardcoded location as requested
        analysis["location"] = "Interstate-5 junction with Highway 42"
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing with OpenAI: {e}")
        # Return default values if analysis fails
        return {
            "location": "Interstate-5 junction with Highway 42",
            "terrain": "Unknown",
            "size": "Unknown",
            "fuel_type": "Unknown",
            "smoke_height": "Unknown",
            "stage": "Unknown"
        }

def extract_field(text, field_name, default_value):
    """Helper function to extract field values from text"""
    import re
    pattern = rf"{field_name}[:\s]+(.*?)(?:\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default_value

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
        message += f"Fire stage is {fire_details['stage']}. "
        message += "Immediate response is needed. "
        message += f"Fuel type: {fire_details['fuel_type']}."

        print(message)
        
        # Actual Twilio call code (commented for testing)
        try:
            call = client.calls.create(
                to=EMERGENCY_NUMBER,
                from_=TWILIO_SMS_NUMBER,
                twiml=f'<Response><Say>{message}</Say></Response>'
            )
            return call.sid
        except Exception as twilio_error:
            print(f"Twilio call error: {twilio_error}")
            return "ERROR_CALL_SID"

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
        message_body += f"Terrain: {fire_details['terrain']}\n"
        message_body += f"Size: {fire_details['size']}\n"
        message_body += f"Fuel Type: {fire_details['fuel_type']}\n"
        message_body += f"Smoke Height: {fire_details['smoke_height']}\n"
        message_body += f"Stage: {fire_details['stage']}"
        
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
        
        # Save the uploaded file
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Print the path to debug
        print(f"Saved file to: {filepath}")
        print(f"File exists: {os.path.exists(filepath)}")
        
        # Analyze image with your model
        confidence, fire_detected = predict_fire(filepath)
        
        # Store results in session
        session['image_path'] = filepath
        session['fire_detected'] = fire_detected
        session['confidence'] = float(confidence)
        
        if fire_detected:
            # Get detailed analysis from OpenAI
            fire_details = analyze_fire_with_openai(filepath)
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
        
        # Fix the image path for display
        try:
            if image_path and image_path.startswith('static/'):
                relative_image_path = image_path
            else:
                relative_image_path = image_path.replace('\\', '/') if image_path else ''
            
            # If the path doesn't start with 'static/' but needs to, add it
            if relative_image_path and not relative_image_path.startswith('static/'):
                relative_image_path = relative_image_path
                
            print(f"Original image path: {image_path}")
            print(f"Processed image path for template: {relative_image_path}")
        except Exception as path_error:
            print(f"Error processing image path: {path_error}")
            relative_image_path = image_path
        
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