BABY CRY TRANSLATOR USING AI/ML

# Project Overview:
Baby Cry Translator is a machine learning–based application that analyzes infant cry audio signals and predicts the possible reason for the cry. Since babies cannot verbally communicate, crying is their primary mode of expression. This project aims to provide caregivers with an **AI-assisted, informational tool** to help interpret baby cries responsibly.

# Problem Statement:
Interpreting the reason behind a baby’s cry is challenging, especially for new parents. Misinterpretation may lead to delayed care and increased anxiety. There is a need for a system that can analyze baby cry sounds and provide an initial understanding of possible causes.

 # Objective:
 - To classify baby cry audio into predefined categories using machine learning  
 - To provide confidence-based predictions through a user-friendly interface  
 - To demonstrate responsible AI usage in a real-world application  

# Approach:
This project follows a supervised machine learning classification approach.

1. Labeled baby cry audio files are organized into categories.
2. Audio features are extracted and converted into numerical form.
3. A Random Forest Classifier is trained on the extracted features.
4. The trained model predicts the probable reason for a new cry audio.
5. Predictions and confidence scores are displayed through a web application.

# Model Used - Random Forest Classifier

# Dataset:
- Audio format: `.wav`
- Categories:
  - Hunger
  - Belly Pain
  - Burping
  - Discomfort
  - Fear
  - Sleepiness
- Dataset is used for academic and learning purposes only.

# Tools and technologies used:
Programming Langauge: Python
ML Library: Scikit-learn
Web Framework: FLASK
Data Handling : Numpy, Pandas
Frontend: HTML, CSS
Development Environment: Jupyter Notebook, VS code
Version Control: Git and Github

# Features:
- Displays top predicted reasons
- Confidence-based prediction bars
- Ethical AI disclaimer for responsible usage

# Evaluation & Results:
The model provides probabilistic predictions with confidence scores.
- Performs well on distinct cry patterns.
- Some overlap exists between similar cry types due to limited dataset size.

# Ethical Considerations:
- Dataset may not represent all infants or environments.
- Predictions are not medical diagnoses.
- Clear disclaimer included to encourage responsible AI usage.
- Users are advised to consult healthcare professionals when needed.

# Future Enhancements:
- Use of larger and more diverse datasets
- Integration of deep learning models
- Real-time cry detection
- Mobile application deployment
