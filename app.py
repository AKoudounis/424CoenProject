from flask import Flask, request, render_template, redirect, url_for, session
import os
import pandas as pd
from google.cloud import firestore, storage
from google.cloud import aiplatform
import logging
from dotenv import load_dotenv
import io

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Initialize Firestore
db = firestore.Client()

# Initialize Cloud Storage client
storage_client = storage.Client()  # Initialize the storage client

# Constants for Vertex AI
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')  # Render the upload form

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        
        if file and file.filename.endswith('.csv'):
            # Upload the file directly to the Cloud Storage bucket
            blob = storage_client.bucket(BUCKET_NAME).blob(file.filename)
            blob.upload_from_file(file)

            # Process the file (pass the Cloud Storage filename)
            process_file(blob.name)  # Use the blob name (file name)

            # Store the filename in session
            session['last_uploaded_file'] = file.filename

            # Redirect to a page with a button to fetch results
            return redirect(url_for('results_button'))
        else:
            return 'Invalid file type', 400
    except Exception as e:
        logging.error(f"Error in upload_file: {e}")
        return f"Error: {e}", 500

def process_file(filename):
    try:
        # Access the file from Cloud Storage
        blob = storage_client.bucket(BUCKET_NAME).blob(filename)
        file_contents = blob.download_as_text()

        # Process the CSV contents directly (no need for local saving)
        df = pd.read_csv(io.StringIO(file_contents))

        # Ensure the 'Date' column is in the correct format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce').dt.strftime('%Y-%m-%d')

        # Prepare instances for prediction
        instances = []

        for index, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['filename'] = filename  # Store filename
            instances.append({
                "TransactionID": row_dict['TransactionID'],
                "Amount": str(row_dict['Amount']),
                "Date": row_dict['Date'],
                "Time": row_dict['Time'],
                "Location": row_dict['Location'],
            })

        # Call the Vertex AI model for predictions
        model = aiplatform.Endpoint(ENDPOINT_ID)
        prediction = model.predict(instances=instances)

        # Log the prediction response for debugging
        logging.info(f"Prediction response: {prediction}")

        # Store results in Firestore
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict['filename'] = filename  # Store filename

            # Safely access 'is_fraud' from predictions
            if 'predictions' in prediction and index < len(prediction.predictions):
                is_fraud = prediction.predictions[index].get('is_fraud', False)  # Default to False if not present
            else:
                is_fraud = False  # Default value if prediction is missing

            row_dict['is_fraud'] = is_fraud  # Add prediction to the row

            # Add to Firestore
            db.collection('transactions').add(row_dict)

        logging.info(f"Processed file '{filename}' and stored results in Firestore.")
    except Exception as e:
        logging.error(f"Error in process_file: {e}")
        raise  # Re-raise the exception for handling in the caller

@app.route('/results_button')
def results_button():
    return render_template('results_button.html')

@app.route('/fetch_results', methods=['GET'])
def fetch_results():
    try:
        # Get the last uploaded filename from session
        last_uploaded_file = session.get('last_uploaded_file')
        if not last_uploaded_file:
            return "No recent uploads found.", 404
        
        page = request.args.get('page', 1, type=int)
        per_page = 100
        offset = (page - 1) * per_page

        # Query Firestore for results matching the last uploaded filename
        docs = db.collection('transactions').where('filename', '==', last_uploaded_file).offset(offset).limit(per_page).stream()

        results = []
        for doc in docs:
            data = doc.to_dict()
            results.append((
                data.get('TransactionID', 'N/A'),
                data.get('Amount', 0),
                data.get('Date', 'N/A'),
                data.get('Location', 'N/A'),
                data.get('Time', 'N/A'),
                data.get('is_fraud', False)
            ))

        # Simulate total_pages for pagination
        total_docs = len(list(db.collection('transactions').where('filename', '==', last_uploaded_file).stream()))  # Get total documents
        total_pages = total_docs // per_page + (1 if total_docs % per_page > 0 else 0)

        return render_template('results.html', results=results, page=page, total_pages=total_pages)
    except Exception as e:
        logging.error(f"Error in fetch_results: {e}")
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
