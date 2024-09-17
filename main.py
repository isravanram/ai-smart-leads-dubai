################################################################################################################################################################################
# MAIN.PY : Entry point for the execution of the App
################################################################################################################################################################################



import os
from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
from werkzeug.utils import secure_filename
import pandas as pd
from ml_pipeline import SmartLeads
import asyncio
from email_scheduler import automated_email_sequence
from db_functions import add_engagement,create_client
from datetime import datetime

# Initializing the Swagger and Flask app
app = Flask(__name__)
swagger = Swagger(app) 

# Creating an instance of the ML model
smart_leads = SmartLeads()

# Create the folder for uploaded files
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  

# Experimental Zone for SME Client Type Classification
@app.route('/smart_lead_predictor', methods=['POST'])
def smart_lead_predictor():
    """
    Predict lead type based on SME data.
    ---
    parameters:
      - name: data
        in: body
        required: true
        schema:
          type: object
          properties:
            sme_name:
              type: string
              description: The name of the SME.
            sme_description:
              type: string
              description: Description of the SME.
            sme_email:
              type: string
              description: Email of the SME.
            sme_size:
              type: integer
              description: Size of the SME (number of employees).
            sme_revenue:
              type: integer
              description: Revenue of the SME.
    responses:
      200:
        description: Lead type prediction result
        schema:
          type: object
          properties:
            lead_type:
              type: string
              description: Type of the lead (Cold, Warm, or Hot).
      400:
        description: Missing data
      500:
        description: Unexpected error
    """
    try:
        # Get data from the request
        data = request.json
        sme_name = data.get('sme_name')
        sme_description = data.get('sme_description')
        sme_email = data.get('sme_email')
        sme_size = data.get('sme_size')
        sme_revenue = data.get('sme_revenue')
        
        # Check for missing data
        if not all([sme_name, sme_description, sme_email, sme_size, sme_revenue]):
            return jsonify({'error': 'Missing data'}), 400

        # Predict lead type
        result = smart_leads.predict_leads_type(sme_name, sme_description, sme_email, sme_size, sme_revenue)
        
        # Return the result
        return jsonify({'lead_type': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# AI Lead generator job
@app.route('/api/ai_lead_generator', methods=['POST'])
async def ai_lead_generator():
    """
    Upload a file and read CSV content to analyze and target potential clients
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The file to upload
    responses:
      200:
        description: File uploaded and read successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: File uploaded and read successfully
                data:
                  type: object
                  additionalProperties: true
      400:
        description: No file part in the request or no file selected
      500:
        description: Failed to upload or read file
    """
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the user actually selected a file
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file:
        # Save the file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            required_columns = ['Company Name', 'Description', 'Email ID', 'Company Size', 'Company Revenue', 'Location']
            email_tasks = []

            # Check for required columns
            if not all(col in df.columns for col in required_columns):
                return jsonify({"message": "Missing required columns in the file"}), 400

            # Iterate over rows and process
            for index, row in df.iterrows():
                if any(loc in str(row['Location']).upper() for loc in ['DUBAI', 'ABU DHABI', 'UAE']):
                    lead_type = smart_leads.predict_leads_type(
                        sme_name=row['Company Name'],
                        sme_description=row['Description'],
                        sme_email=row['Email ID'],
                        sme_size=row['Company Size'],
                        sme_revenue=row['Company Revenue']
                    )
                    
                    if lead_type.upper() in ['WARM LEAD', 'HOT LEAD']:
                        create_client({
                            'email_id': row['Email ID'],
                            'description': row['Description'],
                            'size': row['Company Size'],
                            'revenue': row['Company Revenue']
                        })
                        email_tasks.append(automated_email_sequence(smart_leads,row))
                else:
                    print(f'Enterprise {row["Company Name"]} isn\'t located in the targeted region')

            # Wait for all email tasks to complete
            await asyncio.gather(*email_tasks)
            
            data = df.to_dict(orient='records')
            return jsonify({"message": "File uploaded and read successfully", "data": data}), 200
        
        except Exception as e:
            return jsonify({"message": "Failed to read file", "error": str(e)}), 500

    return jsonify({"message": "Failed to upload file"}), 500


# Responsible for tracking the client engagement
@app.route('/track_click')
def track_click():
    try:
        email = request.args.get('email')

        # Check if the email is provided
        if not email:
            return 'Email parameter is missing', 400 

        # Record the click event
        record = {
            'email_id': email,
            'timestamp': datetime.utcnow()  
        }
        add_engagement(record) 
        return render_template('welcome.html')
    except Exception as e:
        print(f'Error occurred while tracking the client engagement: {e}')
        return 'An error occurred!', 500  


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
