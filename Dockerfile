# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV FLASK_APP=main.py

ENV DB_USERNAME=lead_gen_user
ENV DB_PASSWORD=LeadGen@2024
ENV DATABASE=LeadDataCluster
ENV UPLOAD_FOLDER=dataset
ENV EMAIL_SENDER=sravs.dxb@gmail.com
ENV SMTP_PASSWORD=tbdb mppe trbl lwje


# CMD ["setup_and_run.bat"] # Optional way

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
