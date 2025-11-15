# Corporate Fire Risk Prediction System 

This project is a full-stack web application designed to predict fire risk in a corporate environment using a machine learning model. The system provides a real-time risk assessment based on sensor data input from the user.

## 📸 Deployed Output
Here is a screenshot of the application's user interface, showing a sample prediction result.

### ▫Sensor And Risk Assessment 
<img width="800" alt="image" src="https://github.com/Armaanmd/Corporate-Fire-Risk-Prediction-System/blob/main/Deployed%20system%20screenshot/Screenshot%202025-08-17%20155715.png" />

*Description*: This image shows the main interface where users can input real-time sensor data, including temperature, smoke level, and CO₂. The responsive design ensures the form is user-friendly on both desktop and mobile devices.

### ▫Low-Risk Prediction
<img width="800" alt="image" src="https://github.com/reddy1824/Corporate-Fire-Risk-Perdition-system/blob/main/Model%20Screenshort/Screenshot%202025-11-15%20155941.png"/>

*Description*:This screenshot demonstrates a scenario with low fire risk. The visualization and probability score give a clear "safe" indication, while the input data is presented for transparency and validation.

### ▫High-Risk Prediction
<img width="800" alt="https://github.com/reddy1824/Corporate-Fire-Risk-Perdition-system/blob/main/Model%20Screenshort/Screenshot%202025-11-15%20160017.png" />

*Description*: This image displays a high-risk scenario. The UI immediately communicates the danger with a visual risk indicator that changes color, providing an urgent alert to the user.


## ✨ Features

*●	Real-time Prediction:* Uses a trained RandomForestClassifier to provide instant fire risk predictions.

*●	Intuitive Interface:* A clean, single-page web application that is responsive and easy to use.

*●	Risk Visualization:* A dynamic SVG-based visualization that displays the fire probability and risk level.

*●	Self-Contained Backend:* The backend server handles model loading, prediction, and API communication.

*●	Model Management:* Includes a dedicated endpoint to retrain the machine learning model with new data.

## 💻 Technologies Used

*●	Frontend:* HTML5, CSS3, JavaScript

*●	Backend:* Python, Flask

*●	Machine Learning:* scikit-learn, pandas, numpy

*●	Dependencies Management:* pip, requirements.txt

*●	Deployment & Version Control:* Git, GitHub

## 🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

## Prerequisites
You need to have Python 3 installed. You can download it from the official Python website.

## Step 1: Clone the Repository
git clone [https://github.com/reddy1824/Corporate-Fire-Risk-Perdition-system]

cd your-repository-name

(Note: Remember to replace the URL with your own GitHub repository URL)

## Step 2: Set up the Backend
Navigate to the backend directory and install the required Python libraries.

cd backend
pip install -r requirements.txt

## Step 3: Run the Backend Server
Start the Flask server. This will train the model and make the prediction API available.

python app.py

The server will run on http://127.0.0.1:5000.

## Step 4: Run the Frontend
The frontend is a static HTML file. Simply open the index.html file in your preferred web browser.

You can now use the application by entering sensor data and clicking "Predict Risk." The frontend will send the data to the backend, which will return the prediction result.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

* *Mohan Reddy N* - [www.linkedin.com/in/
mohan-reddy-n-b9b974395]
