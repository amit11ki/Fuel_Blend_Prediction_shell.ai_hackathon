import sys
from pathlib import Path
import uuid
# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import google.generativeai as genai
from config import GEMINI_API_KEY
from src.prompts import SYSTEM_INSTRUCTION
from src.predictor import run_central_predictor
from werkzeug.utils import secure_filename
import os
import pandas as pd
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully.")
except ValueError as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise

# Create the model with system instruction
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

try:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    logger.info("Generative model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize generative model: {str(e)}")
    raise

UPLOAD_FOLDER = 'Uploads'
PREDICTION_FOLDER = 'predictions'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Temporary storage for prediction inputs
prediction_inputs = {}

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/predictor', methods=['GET'])
def predictor():
    return render_template('predictor.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'No message provided.'}), 400

    logger.debug(f"Received chat message: {user_message}")
    try:
        response = model.generate_content(user_message, stream=False)
        ai_response = response.text
        logger.debug(f"Generated response: {ai_response}")
        return jsonify({'response': ai_response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'response': f'Error generating response: {str(e)}'}), 500

@app.route('/load_sample', methods=['GET'])
def load_sample():
    try:
        test_csv_path = os.path.join(project_root, 'data', 'test.csv')
        df = pd.read_csv(test_csv_path)
        if len(df) < 2:
            return jsonify({'response': 'Test CSV has fewer than 2 rows.'}), 400
        # Get the second row (index 1) and exclude the ID column
        sample_row = df.iloc[1].to_dict()
        sample_row.pop('ID', None)
        return jsonify(sample_row)
    except Exception as e:
        logger.error(f"Load sample error: {str(e)}")
        return jsonify({'response': f'Failed to load sample data: {str(e)}'}), 500

def create_modern_table_html(df):
    """Create a modern styled table HTML from DataFrame"""
    if df.empty:
        return '<p class="text-slate-400">No data to display</p>'
    
    # Start table with glassmorphism styling
    html = '<div class="overflow-x-auto max-w-full mt-4">'
    html += '<table class="min-w-full bg-[#14141c] border border-[#2a2a3a] rounded-lg">'
    
    # Create header
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th class="px-6 py-3 bg-[#1e293b] text-left text-sm font-semibold text-slate-300 uppercase">{col}</th>'
    html += '</tr></thead>'
    
    # Create body
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr class="hover:bg-[#1e293b] transition-colors">'
        for col in df.columns:
            value = row[col]
            # Format numbers nicely
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)
            html += f'<td class="px-6 py-4 border-t border-[#2a2a3a] text-sm text-slate-300">{formatted_value}</td>'
        html += '</tr>'
    html += '</tbody>'
    
    html += '</table>'
    html += '</div>'
    
    return html

@app.route('/start_predict', methods=['POST'])
def start_predict():
    if 'file' not in request.files:
        return jsonify({'response': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'response': 'No selected file.'}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'predict_input_{session_id}.csv')
        file.save(input_path)
        prediction_inputs[session_id] = input_path
        return jsonify({'session_id': session_id})
    
    return jsonify({'response': 'Invalid file. Please upload a CSV file.'}), 400

@app.route('/predict_stream/<session_id>', methods=['GET'])
def predict_stream(session_id):
    if session_id not in prediction_inputs:
        return Response(f"data: {json.dumps({'error': 'Invalid session ID'})}\n\n", mimetype='text/event-stream')

    input_path = prediction_inputs[session_id]
    output_filename = f'predictions_all_blends_{session_id}.csv'
    output_path = os.path.join(app.config['PREDICTION_FOLDER'], output_filename)
    
    def generate_progress():
        try:
            df = None
            for progress_data in run_central_predictor(input_path, stream_progress=True):
                if 'error' in progress_data:
                    yield f"data: {json.dumps({'error': progress_data['error']})}\n\n"
                elif progress_data.get('progress') == 100 and 'result' in progress_data:
                    df = pd.DataFrame(progress_data['result'])
                    df.to_csv(output_path, index=False)
                    # Show all rows if <= 10, else top 10
                    preview_df = df if len(df) <= 10 else df.head(10)
                    yield f"data: {json.dumps({
                        'progress': 100,
                        'blend': 'Complete',
                        'preview': create_modern_table_html(preview_df),
                        'download_url': f'/download/{output_filename}',
                        'total_predictions': len(df)
                    })}\n\n"
                else:
                    yield f"data: {json.dumps(progress_data)}\n\n"
            # Clean up
            if os.path.exists(input_path):
                os.remove(input_path)
            prediction_inputs.pop(session_id, None)
            if df is None:
                yield f"data: {json.dumps({'error': 'Prediction failed for one or more blends.'})}\n\n"
        except Exception as e:
            logger.error(f"CSV prediction stream error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Prediction failed: {str(e)}'})}\n\n"

    return Response(generate_progress(), mimetype='text/event-stream')

@app.route('/start_predict_single', methods=['POST'])
def start_predict_single():
    data = request.json
    if not data:
        return jsonify({'response': 'No data provided.'}), 400

    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        # Save input data temporarily
        df = pd.DataFrame([data])
        df['ID'] = 1  # Add a dummy ID for compatibility with predictors
        temp_input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_single_input_{session_id}.csv')
        df.to_csv(temp_input_path, index=False)
        prediction_inputs[session_id] = temp_input_path
        return jsonify({'session_id': session_id})
    except Exception as e:
        logger.error(f"Start single prediction error: {str(e)}")
        return jsonify({'response': f'Failed to start prediction: {str(e)}'}), 500

@app.route('/predict_single_stream/<session_id>', methods=['GET'])
def predict_single_stream(session_id):
    if session_id not in prediction_inputs:
        return Response(f"data: {json.dumps({'error': 'Invalid session ID'})}\n\n", mimetype='text/event-stream')

    temp_input_path = prediction_inputs[session_id]
    
    def generate_progress():
        try:
            for progress_data in run_central_predictor(temp_input_path, stream_progress=True):
                if 'error' in progress_data:
                    yield f"data: {json.dumps({'error': progress_data['error']})}\n\n"
                else:
                    yield f"data: {json.dumps(progress_data)}\n\n"
            # Clean up
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            prediction_inputs.pop(session_id, None)
        except Exception as e:
            logger.error(f"Single prediction stream error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Prediction failed: {str(e)}'})}\n\n"

    return Response(generate_progress(), mimetype='text/event-stream')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['PREDICTION_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # Run Flask with debug mode but disable reloader to prevent restarts during predictions
    app.run(debug=True, use_reloader=False)