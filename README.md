# Amazon Reviews-Sentiment Analysis
 
## **Requirements**

Python 3.8+
PyCharm (or any preferred IDE)
Conda

## **Setup Instructions**

### Step 1: Create Conda Environment

1.Open PyCharm. 

2.Open the terminal in PyCharm.

3.Create a new Conda environment with Flask:

    conda create --name sentiment_analysis_env python=3.8
    conda activate sentiment_analysis_env
    pip install flask

### Step 2: Download Dataset

1.Download the text.ft.txt dataset from [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews?resource=download
).

2.Add the text.ft.txt file to your project's root directory.

Step 3: Install Required Packages

1.Ensure you are in the root directory of your project.

2.Run the following command to install the required packages:

    pip install -r requirements.txt



## Running the Application


### Step 1: Train the Model

Before running the web application, you need to train the model for the first time to create all the necessary files. This allows the program to store new information from subsequent trainings.

1.Run the train_model.py script:

    python train_model.py

### Step 2: Start the Web Application

1.Run the app.py script:
`    python app.py
`

Open your web browser and go to **http://127.0.0.1:5000.**

## Usage

Sentiment Analysis Prediction: You can ask the model to give you a sentiment analysis prediction by providing a review. The application will analyze the sentiment of the given review.
Retrain the Model: To retrain the model, click on the "**Train Model**" button on the web page and wait for the training to complete.

## Notes

Make sure the **text.ft.txt** file is present in the root directory before running any scripts.
The initial model training might take some time, depending on your system's performance.
Feel free to contribute or raise issues if you encounter any problems!






