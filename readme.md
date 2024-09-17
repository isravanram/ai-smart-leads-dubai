
# Two ways to execute the App

# A: Direct run:
1) In the terminal run setup_and_run.bat
2) Navigate to the url http://127.0.0.1:8080/apidocs/ to open SwaggerUI
3) Upload the Company profile information in csv format to the post_api_upload_company_profiles API to trigger the Smart Lead AI model
4) You can also use the test_model playground to test the predictions for each client. (Cold, Warm and Hot clients)

# B: Docker run:

# Docker Container Setup and Usage

This guide explains how to build and run a Docker container that executes a batch file to set environment variables and run a Python script.

## Prerequisites

- **Docker**: Install Docker from [Docker's official website](https://www.docker.com/products/docker-desktop).

## Instructions

### 1. Download the Project Files

Ensure you have the following files:
- `Dockerfile`
- `setup_and_run.bat`
- `main.py`

### 2. Open Command Prompt or Terminal

Open Command Prompt (on Windows) or Terminal (on macOS/Linux).

### 3. Navigate to the Project Directory

Change to the directory where you have the project files:

cd path\to\your\project

### 4. Build Docker Image

docker build -t my-windows-app .


### 5. Run the docker container
docker run --name my-running-app my-windows-app
