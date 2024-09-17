################################################################################################################################################################################
# EMAIL_SCHEDULER.PY : Includes Email automation workflow
################################################################################################################################################################################

import os
import time
import asyncio
from email.message import EmailMessage
import ssl
import smtplib
from db_functions import get_latest_engagement_level
from ml_pipeline import SmartLeads

EMAIL_SENDER = os.getenv("EMAIL_SENDER")  
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") 

# Send email using SMTP API
def send_email(to_email, subject, content):
    try:
        print(f'\n---------- Sending email to the recipient: {to_email} ----------')
        sender = EMAIL_SENDER
        receiver = to_email
        password = SMTP_PASSWORD
        body = content
        print(f'Sender: {sender}, Receiver: {receiver}')
        
        # Create an email message object
        em = EmailMessage()
        em['From'] = sender
        em['To'] = receiver
        em['Subject'] = subject
        em.set_content(body)

        # em.set_content("This is an HTML email", subtype='html')
        # em.set_payload(body)

        # Set up the SMTP server and send the email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender, password)
            response = smtp.sendmail(sender, receiver, em.as_string())
            if response:
                print("Failed to send email to the following recipients:")
                for recipient, error in response.items():
                    print(f"{recipient}: {error}")
        print(f'\n---------- Email sent successfully to the following recipient: {to_email}---------- ')

    except smtplib.SMTPAuthenticationError as auth_error:
        print(f'Authentication error: {auth_error}')
    except smtplib.SMTPConnectError as conn_error:
        print(f'Connection error: {conn_error}')
    except smtplib.SMTPRecipientsRefused as recipient_error:
        print(f'Recipient error: {recipient_error}')
    except smtplib.SMTPSenderRefused as sender_error:
        print(f'Sender error: {sender_error}')
    except smtplib.SMTPDataError as data_error:
        print(f'Data error: {data_error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

# Generate email content using Machine Learning
def generate_email(smart_leads,lead_info, engagement_level):
    try:
        subject,email_content = smart_leads.generate_email_content(lead_info,engagement_level)
        send_email(lead_info['Email ID'], subject, email_content)
    except Exception as e:
        print('Error occured while generating email')

# Automate email process
async def automated_email_sequence(smart_leads,lead_info):
    try:
        print(f'\n---------- Email automation job for the client {lead_info['Company Name']} with email id : {lead_info['Email ID']} started ---------- ')
        generate_email(smart_leads,lead_info, engagement_level=0)
        previous_engagement_level = get_latest_engagement_level(lead_info['Email ID'])
        attempts = 0
        while True:
            attempts += 1
            # Used asyncio.sleep instead of time.sleep to avoid blocking the event loop
            await asyncio.sleep(30)  # Introduced a non-blocking delay
            latest_engagement_level = get_latest_engagement_level(lead_info['Email ID'])
            if latest_engagement_level > previous_engagement_level:
                print(f"Engagement level increased to {latest_engagement_level} for the email id: {lead_info['Email ID']}. Sending follow-up email.")
                generate_email(smart_leads,lead_info, latest_engagement_level)
                previous_engagement_level = latest_engagement_level
                break
            else:
                print(f"No response from the client: {lead_info['Email ID']} yet. Current engagement level: {latest_engagement_level}")    
            if attempts > 30:
                print("Client hasn't responded yet.")
                break  # Or send a personalized email again to catch the client's attention
        print(f'\n---------- Email automation job for the client {lead_info['Company Name']} with email id: {lead_info['Email ID']} completed ---------- ')   
    except Exception as e:
        print(f'Error occurred while monitoring the email automation: {e}')
        return 0