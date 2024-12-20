# Spam-Email-Classification


## Overview
This project implements a spam email classifier using Natural Language Processing (NLP) techniques and a Naive Bayes machine learning model. The classifier is designed to distinguish between spam and non-spam emails based on their textual content. The project includes data cleaning, visualization, feature extraction, model training, and a simple graphical user interface (GUI) for user interaction.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [GUI Interaction](#gui-interaction)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Contributing](#contributing)
- [License](#license)

## Features
- Data cleaning and preprocessing of email text.
- Visualization of spam vs. non-spam email distribution.
- Text analysis features such as character count, word count, and sentence count.
- Word cloud generation for visual representation of spam and non-spam emails.
- Naive Bayes classification model for spam detection.
- Interactive GUI for real-time email classification.

## Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `sklearn`
- `wordcloud`
- `ipywidgets`
- `joblib`

You can install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud ipywidgets joblib
```

## Dataset
The dataset used in this project is a CSV file named `spam.csv`, which contains two columns:
1. **Label**: Indicates whether the email is spam or not (1 for spam, 0 for not spam).
2. **Email_Text**: The content of the email.

Ensure that the dataset is placed in the same directory as the script before running it.

## Installation
1. Clone this repository or download the script file.
2. Ensure you have all the required libraries installed as mentioned above.

## Usage
1. Run the script in a Jupyter Notebook or any Python environment that supports `ipywidgets`.
2. The GUI will prompt you to enter an email's text.
3. Click on the "Classify Email" button to get the classification result.

## Model Training
The model is trained using a Multinomial Naive Bayes classifier. The dataset is split into training (80%) and testing (20%) sets to evaluate the model's performance.

## Evaluation
The accuracy of the model is printed after training, providing an indication of how well it performs on unseen data.

## GUI Interaction
The GUI allows users to input an email's text and receive immediate feedback on whether it is classified as spam or not spam. This interaction is facilitated by the `ipywidgets` library.

### Example Usage:
```python
email_text = "Congratulations! You've won a $1000 gift card."
result = classify_email(email_text)
print(result)  # Output: This email is classified as **Spam**.
```

## Saving and Loading the Model
The trained model and vectorizer are saved to disk using `joblib`, allowing for easy loading in future sessions without retraining.

```python
import joblib

# Load the model and vectorizer
naive_bayes_model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
```

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or open an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

Feel free to modify any sections to better fit your project's specifics or personal preferences!
