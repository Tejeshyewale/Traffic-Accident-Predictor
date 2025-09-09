Project Overview

This project is a complete, end-to-end machine learning solution designed to predict the likelihood of a traffic accident based on various environmental, driver, and road-related factors. The goal was to build a full-stack application, from a raw dataset to a live, cloud-hosted web service.
Key Features

    Data Preprocessing: A robust Python script handles missing values (imputation with median/mode) and transforms categorical data using One-Hot and Label Encoding.

    Hybrid Prediction Model: The backend uses a powerful XGBoost Classifier. A custom logic layer was implemented to overcome class imbalance, ensuring the model provides responsive and accurate predictions for high-risk scenarios.

    Web Application: A lightweight web application built with Flask provides a user-friendly interface for inputting parameters and receiving real-time predictions.

    Containerization: The entire application is packaged into a Docker container, making it portable and easy to deploy across different environments.

    Cloud Deployment: The project is deployed as a live web service on AWS Elastic Beanstalk, complete with automated resource provisioning and security group configuration for public access.

Project Structure

The repository is organized to reflect the full development lifecycle:

/traffic_accident_predictor
├── app.py                     # The Flask API and web server
├── Dockerfile                 # Instructions for building the Docker image
├── Procfile                   # Command to run the application on AWS
├── Dockerrun.aws.json         # AWS configuration file for Docker deployment
├── xgboost_traffic_model.pkl  # The trained machine learning model
├── dataset_traffic_accident_prediction1.csv # The raw dataset
└── README.md                  # This file

Getting Started
Prerequisites

Before running the application, ensure you have the following installed:

    Docker Desktop

    Python 3.9 or higher

    A text editor or IDE (e.g., VS Code)

1. Building the Docker Image

Clone this repository to your local machine and navigate to the project's root directory in your terminal.

Then, build the Docker image with the following command:

docker build -t traffic-predictor .

2. Running the Application Locally

Once the image is built, you can run the application in a Docker container. This will start a local server that you can access from your browser.

docker run -p 5000:8000 traffic-predictor

Open your web browser and go to http://localhost:5000 to see the application in action.
Cloud Deployment (AWS)

For a full deployment to the cloud, follow these steps:
1. AWS Setup

    Create an AWS account and an IAM user with AdministratorAccess (for this project).

    Install and configure the AWS CLI and EB CLI on your local machine using the IAM user credentials.

2. Deploy with EB CLI

Run the following commands from your terminal in the project's root directory:

# Initialize the Elastic Beanstalk application
eb init

# Create the environment and deploy your application
eb create traffic-predictor-env

Once the deployment is complete, the EB CLI will provide you with the public URL for your live application.
Contributions

Feel free to open an issue or submit a pull request if you find any bugs or have suggestions for improvement.
