################################################################################################################################################################################
# ML_PIPELINE.PY : Includes model building, prediction, and automated generation of personalized email content using NLP.
################################################################################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import spacy
from textblob import Word,TextBlob

# Loading the English language model 'Spacy' for processing and analyzing text data
nlp = spacy.load("en_core_web_sm")

# Class handling the ML operations
class SmartLeads:
    def __init__(self):
        self.X, self.y, self.vectorizer, self.scaler, self.model = self.build_model()
        self.compute_cross_validation_score(cv=19, scoring="accuracy")

    # Building the model
    def build_model(self):
        try:
            print('==================== Building the Model ====================')
            
            # Load dataset
            data = pd.read_csv('dataset/smart_leads_ai.csv', encoding='ISO-8859-1')

            # Handle missing values in the Description column
            data['Description'] = data['Description'].fillna('')

            # Initialize the TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=200, ngram_range=(1, 2))
            description_tfidf = tfidf_vectorizer.fit_transform(data['Description'])

            # Scale the numerical columns
            scaler = StandardScaler()
            numerical_data = data[['Company Size', 'Company Revenue (in million)']]
            scaled_numerical_data = scaler.fit_transform(numerical_data)

            # Combine preprocessed data
            X_preprocessed = np.hstack((scaled_numerical_data, description_tfidf.toarray()))

            # Target variable
            y = data['Leads Score'].values

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

            # Apply SMOTE to balance the classes in the training data
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            # Initialize the RandomForest model with class_weight set to 'balanced'
            rf_model = RandomForestClassifier(class_weight='balanced', random_state=39)

            # Train the model
            rf_model.fit(X_train_balanced, y_train_balanced)

            # Make predictions
            y_pred = rf_model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            # Print results
            print("Accuracy of the Random Forest model:", accuracy)
            print("Classification Report:\n", classification_rep)
            print('==================== Model built ====================')
            return X_preprocessed, y, tfidf_vectorizer, scaler, rf_model

        except FileNotFoundError as file_error:
            print(f'File not found: {file_error}')
        except pd.errors.EmptyDataError as empty_data_error:
            print(f'Empty data error: {empty_data_error}')
        except KeyError as key_error:
            print(f'Key error: {key_error}')
        except ValueError as value_error:
            print(f'Value error: {value_error}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')

    # Computing the cross validation score
    def compute_cross_validation_score(self, cv, scoring='accuracy'):
        try:
            print('\n\n==================== Evaluating the Model ====================')
            # Perform cross-validation
            accuracies = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring)
            # Calculate the average accuracy
            average_accuracy = accuracies.mean()
            print(f'Cross-Validation Accuracies: {accuracies}')
            print(f'Average Accuracy: {average_accuracy}')
            print('\n\n==================== Model evaluated ====================')
        except ValueError as value_error:
            print(f'Value error occurred during cross-validation: {value_error}')
        except TypeError as type_error:
            print(f'Type error occurred during cross-validation: {type_error}')
        except Exception as e:
            print(f'An unexpected error occurred during cross-validation: {e}')

    # This is for sample testing
    def check_sample_lead_predictions(self):
        try:
            # Hardcoded new input data
            print('\n\n==================== Checking Sample Predictions ====================')
            descriptions_list = [
                'A consulting based business in Dubai focusing on the latest advancements.',
                'A full-service healthcare center known for its comprehensive range of medical specialties, including pediatrics, oncology, and reproductive health.',
                'We are committed to providing an exceptional beauty experience that leaves our clients feeling confident and rejuvenated. Our team of highly trained stylists and beauty experts specializes in a wide range of services, from precision haircuts and vibrant color treatments to luxurious facials, manicures, and professional makeup applications.',
                'AI-driven automation with the primary focus to improve decision-making processes.',
                'The clinic provides routine check-ups, advanced diagnostics, and a range of treatments to support the health and well-being of patients from all walks of life, with a strong emphasis on being a comprehensive healthcare center'
            ]
            company_size_list = [62, 25, 33, 33, 37]
            company_revenue_list = [38, 9, 18, 18, 42]
            
            for index in range(len(descriptions_list)):
                description = descriptions_list[index]
                company_size = company_size_list[index]
                company_revenue = company_revenue_list[index]
                new_description_tfidf = self.vectorizer.transform([description])
                print('\n==================== Enterprise info ====================')
                print(f'Description : {description}')
                print(f'Company Size : {company_size}')
                print(f'Company Revenue : {company_revenue}')
                new_numerical_data = np.array([[company_size, company_revenue]])
                new_scaled_numerical_data = self.scaler.transform(new_numerical_data)
                new_X_preprocessed = np.hstack((new_scaled_numerical_data, new_description_tfidf.toarray()))
                new_predictions = self.model.predict(new_X_preprocessed)
                print(f"Predictions for the new input: {new_predictions}\n")
        except AttributeError as attr_error:
            print(f'Attribute error occurred: {attr_error}')
        except ValueError as value_error:
            print(f'Value error occurred while processing new predictions: {value_error}')
        except TypeError as type_error:
            print(f'Type error occurred while processing new predictions: {type_error}')
        except Exception as e:
            print(f'An unexpected error occurred while processing new predictions: {e}')

    # Predicting the Leads type (Hot Lead, Warm Lead, Cold Lead)
    def predict_leads_type(self, sme_name, sme_description, sme_email, sme_size, sme_revenue):
        try:
            print('\n==================== Predicting Lead type ====================')
            # Transform the new description using the vectorizer
            new_description_tfidf = self.vectorizer.transform([sme_description])
            
            # Print enterprise information
            print('Enterprise name : {0}'.format(sme_name))
            print('Description : {0}'.format(sme_description))
            print('Company Size : {0}'.format(sme_size))
            print('Company Revenue : {0}'.format(sme_revenue))
            
            # Prepare numerical data
            new_numerical_data = np.array([[sme_size, sme_revenue]])
            new_scaled_numerical_data = self.scaler.transform(new_numerical_data)
            
            # Combine scaled numerical data with TF-IDF features
            new_X_preprocessed = np.hstack((new_scaled_numerical_data, new_description_tfidf.toarray()))
            
            # Make predictions
            new_predictions = self.model.predict(new_X_preprocessed)
            if str(new_predictions[0]) == '0':
                prediction =  'Cold Lead'
            elif str(new_predictions[0]) == '1':
                prediction =  'Warm Lead'
            else:
                prediction = 'Hot Lead'
            print(f'==================== Prediction: {prediction}====================')
            # Return lead type based on predictions
            return prediction

        except AttributeError as attr_error:
            print(f'Attribute error: {attr_error}')
        except ValueError as val_error:
            print(f'Value error: {val_error}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
    
    # Detecting the industry type
    def detect_industry(self, text):
        try:
            industries = ["consulting", "healthcare", "technology", "beauty", "retail", "finance", "education"]
            doc = nlp(text)
            industry_detected = [industry for industry in industries if industry in text.lower()]
            return industry_detected if industry_detected else ["Industry Sector"]
        except Exception as e:
            print(f'Error occurred while detecting industry: {e}')
            return ["Error detecting the industry"]

    # Detecting the pain points for the industry
    def detect_pain_points(self, text):
        try:
            doc = nlp(text)
            pain_keywords = ["struggling", "issue", "problem", "challenge", "difficulty", "abandonment"]
            pain_points = [token.text for token in doc if token.lemma_ in pain_keywords]
            return pain_points if pain_points else ["AI business hurdles"]
        except Exception as e:
            print(f'Error occurred while detecting pain points: {e}')
            return ["Error detecting the pain points"]


    # Generate Email content using NLP
    def generate_email_content(self, lead_info, engagement_level):
        try:
            # Detect industry and pain points from the description
            industry = self.detect_industry(lead_info['Description'])
            pain_points = self.detect_pain_points(lead_info['Description'])

            # Base email content for all engagement levels
            email_content = f"""
                Dear {lead_info['Company Name']},

                We noticed that you are in the {industry[0]} industry and might be facing challenges like {', '.join(pain_points)}.

                We can help address these pain points with our tailored solutions.

                Let's connect and explore how we can assist your business!

                Click here to explore our solutions: http://127.0.0.1:8080/track_click?email={lead_info['Email ID']}

                Best regards,

                The Taippa Team
                United Arab Emirates
                Ph: +971-77777000000
            """


            # Add a personalized follow-up if engagement level is greater than 0
            if engagement_level > 0:
                personalized_message = self.create_personalized_message(lead_info, engagement_level)
                email_content = f"{personalized_message}"
            
            # Set the subject based on engagement level
            subject = "Discover Innovative AI Strategies for Your Business" if engagement_level == 0 else "Following Up on AI Strategies: Next Steps for Your Business"
            
            return subject, email_content

        except KeyError as e:
            print(f"KeyError: Missing expected key in lead_info: {e}")
            raise

        except IndexError as e:
            print(f"IndexError: Issue with accessing industry or pain points: {e}")
            raise

        except Exception as e:
            print(f"An unexpected error occurred in generate_email_content: {e}")
            raise

    # Creating personalized email for follow up (Can be enhanced in future to include NLP to rephrase content, engage with client's response and analyze via sentiment analysis to send personalized messages. Currently implemented with (less) dummy data)
    def create_personalized_message(self, lead_info, engagement_level):
        try:
            # Add personalized follow-up content
            message = f"""
                Dear {lead_info['Company Name']},

                I hope this email finds you well.

                I wanted to follow up on our recent discussion regarding AI solutions for your business. We noticed your interest and would like to offer further assistance in exploring how our AI solutions can drive innovation and efficiency in your operations.

                Here’s a brief overview of what we can offer:
                A) Tailored AI Solutions
                B) Expert Consultation
                C) Case Studies & Success Stories

                We’re committed to helping you leverage AI to achieve your business goals. If you have any questions or would like to schedule a demo or consultation, please feel free to reach out.

                Best regards,

                The Taippa Team
                United Arab Emirates
                Ph: +971-77777000000
            """

            return message

        except Exception as e:
            print(f"An unexpected error occurred in create_personalized_message: {e}")
            raise
