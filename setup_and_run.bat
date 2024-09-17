@echo off

:: Set environment variables
set DB_USERNAME=lead_gen_user
set DB_PASSWORD=LeadGen@2024
set DATABASE=LeadDataCluster
set UPLOAD_FOLDER=dataset
set EMAIL_SENDER=sravs.dxb@gmail.com
set SMTP_PASSWORD=tbdb mppe trbl lwje

:: Run the Python script
python main.py
