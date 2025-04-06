# Phishing Email Detection Algorithm

A machine learning application built with Streamlit that uses ensemble learning (soft voting) to detect phishing emails with high accuracy.

## Features

- **Single Email Analysis**: Analyze individual emails for phishing attempts
- **Batch Processing**: Upload CSV files for analyzing multiple emails at once
- **Spam Folder Management**: Automatically moves suspected phishing emails to a spam folder
- **Model Evaluation**: View performance metrics to understand the model's accuracy
- **History Tracking**: Keep a record of all email predictions

## Technologies Used

- Python 3.x
- Streamlit for the web interface
- Scikit-learn for machine learning models:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Ensemble Voting Classifier
- Pandas for data processing
- Matplotlib for data visualization

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/phishing-email-detection.git
cd phishing-email-detection
```

2. Install required packages:
```
pip install streamlit pandas scikit-learn numpy matplotlib joblib
```

3. Run the application:
```
streamlit run soft_vote_app.py
```

## Usage

### Single Email Analysis
1. Enter the sender email, subject, and keywords
2. Toggle URL and attachment options if applicable
3. Enter the timestamp
4. Click "predict" to analyze the email

### Batch Processing
1. Prepare a CSV file with columns: sender, subject, keywords
   (Optional columns: has_url, has_attachment, timestamp)
2. Upload the CSV file through the "Upload CSV" section
3. View the analysis results in the table

### Viewing Results
- Check the Model Evaluation section to see accuracy, precision, recall, and F1-score
- Browse your prediction history and spam folder

## Dataset

The model is trained on a custom phishing dataset that includes various email features and labeled examples of legitimate and phishing emails.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

© Copyright 2025. Designed by Edwin and Jael. 