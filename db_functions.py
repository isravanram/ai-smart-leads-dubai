################################################################################################################################################################################
# DB_FUNCTIONS.PY : Consists of DB related operations to store Client info and interaction details 
################################################################################################################################################################################

import os
from pymongo import MongoClient, errors
from datetime import datetime
from urllib.parse import quote_plus

# DB creds for MongoDB
DB_USERNAME = os.getenv("DB_USERNAME")  
DB_PASSWORD = os.getenv("DB_PASSWORD") 
DATABASE = os.getenv("DATABASE")  

# Initialize the MongoDB client and database
def get_database():
    encoded_username = quote_plus(DB_USERNAME)
    encoded_password = quote_plus(DB_PASSWORD)
    uri = f'mongodb+srv://{encoded_username}:{encoded_password}@leaddatacluster.falji.mongodb.net/?retryWrites=true&w=majority&appName=LeadDataCluster'
    client = MongoClient(uri)
    db = client[DATABASE]  # Database
    return db

# Create client records
def create_client(client_data):
    
    try:
        print(f'\n Adding the client details to the db having email_id {client_data['email_id']}')
        db = get_database()
        try:
            collection = db.create_collection('client')
        except errors.CollectionInvalid:
            collection = db['client']
            print("Collection named 'client' already exists, skipping creation.")

        try:
            collection.create_index('email_id', unique=True)
        except errors.DuplicateKeyError:
            print("Unique index on 'email' already exists.")
        
        email_id = client_data['email_id']
        result = collection.update_one({'email_id': email_id}, {'$set': client_data}, upsert=True)
        if result.matched_count:
            print(f"Updated client record with email_id: {email_id}.")
        else:
            print(f"Inserted new client record with email_id: {email_id}.")
        print(f'\n Updated the client details to the db having email_id {client_data['email_id']}')
    except Exception as e:
        print('Error occured while adding client details to the database: {0}'.format(e))
    

# Create client email engagement records 
def add_engagement(engagement_info):
    try:
        print(f'\nAdding mail engagement details for the client: {engagement_info['email_id']}')
        db = get_database()
        try:
            collection = db.create_collection('engagement')
        except errors.CollectionInvalid:
            collection = db['engagement']
            print("Collection named 'engagement' already exists, skipping creation.")

        try:
            collection.create_index('email_id', unique=True)
        except errors.DuplicateKeyError:
            print("Unique index on 'email_id' already exists.")
            
        result = collection.update_one(
            {'email_id': engagement_info['email_id']},
            {'$set': engagement_info},
            upsert=True
        )
        
        if result.matched_count:
            collection.update_one(
                {'email_id': engagement_info['email_id']},
                {'$inc': {'engagement_level': 1}}
            )
            print(f"Engagement info updated for email_id: {engagement_info['email_id']}.")
        else:
            print(f"Engagement data added for email_id: {engagement_info['email_id']}.")
        print(f'Updated the mail engagement details for the email_id: {engagement_info['email_id']}')
    except Exception as e:
        print(f'Error occurred while adding client engagement details to the database: {e}')

# Retrieve the latest client interaction details
def get_latest_engagement_level(email_id):
    try:
        print(f'\nFetching the latest engagement level for the email id: {email_id}')
        db = get_database()
        collection = db['engagement']
        
        latest_engagement = collection.find_one(
            {'email_id': email_id},
            sort=[('timestamp', -1)]  
        )

        if latest_engagement:
            engagement_level = latest_engagement.get('engagement_level', None)
            if engagement_level is None:
                print(f"No engagement level found for email_id {email_id}. Initializing to 0.")
                collection.update_one(
                    {'_id': latest_engagement['_id']},  
                    {'$set': {'engagement_level': 0}}  
                )
                engagement_level = 0 
            print(f"Latest engagement level for email_id {email_id}: {engagement_level}")
        else:
            print(f"No engagement data found for the email_id: {email_id}. Creating a new entry with engagement_level 0.")
            # If no record exists for this email_id, create one with engagement_level = 0
            collection.insert_one({
                'email_id': email_id,
                'engagement_level': 0,
                'timestamp': datetime.utcnow()
            })
            engagement_level = 0

        print(f'Fetched the latest engagement level for the email id: {email_id}')
        return engagement_level

    except Exception as e:
        print(f"Error fetching engagement level for {email_id}: {e}")
        return 0 