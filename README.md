# NLP Case Study – DonorsChoose Project Approval Prediction

## Project Overview

This project uses **Natural Language Processing (NLP)** and Machine Learning to predict whether a DonorsChoose project will be approved.

The text data comes from project titles, summaries, and essays. Different NLP techniques and machine learning models are used to classify projects as approved or not approved.

## Dataset

The project uses a preprocessed version of the **DonorsChoose dataset**.

The main text columns used are:

* `cleaned_titles`
* `cleaned_summary`
* `cleaned_essays`

The target column is:

* `project_is_approved`

## Steps Performed

### 1. Exploratory Data Analysis

The dataset was checked for:

* Dataset size and information
* Missing values
* Text columns
* Target class distribution

The target classes were found to be imbalanced.

### 2. Text Preprocessing

The text was prepared using:

* Tokenization
* Lemmatization
* Removing low-frequency words
* Combining the title, summary, and essay into one text column

### 3. Feature Extraction

Two main approaches were explored:

* **TF-IDF**

  * Unigrams
  * Bigrams
  * Trigrams
* **Word2Vec**

  * Custom Word2Vec model
  * Pretrained GloVe embeddings

### 4. Machine Learning Models

The following models were tested:

* Logistic Regression
* Naive Bayes
* Random Forest
* Support Vector Machine (SVM)

Their accuracy was compared to find the better-performing model.

### 5. Hyperparameter Tuning

`GridSearchCV` was used to tune:

* Logistic Regression
* SVM
* Naive Bayes

### 6. Deep Learning Model

A Functional API neural network was created using:

* TF-IDF features
* Word2Vec features
* Dense layers
* Batch Normalization
* Dropout
* Sigmoid output layer

The model was trained to predict whether a project would be approved.

## Model Evaluation

The models were evaluated using:

* Accuracy
* Classification Report
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

## Results

Among the classical machine learning models, **SVM performed the best**.

The neural network combined TF-IDF and Word2Vec features to use both important words and their semantic information.

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Gensim
* Matplotlib
* Seaborn
* TensorFlow / Keras

## How to Run

1. Open the notebook in **Google Colab** or Jupyter Notebook.
2. Install the required Python libraries.
3. Make sure the dataset is available at the required path.
4. Run the notebook cells from top to bottom.

## Future Improvements

Some possible improvements are:

* Train the neural network for more epochs.
* Use Early Stopping.
* Perform more hyperparameter tuning.
* Use a larger and more balanced dataset.
* Experiment with different NLP models and embeddings.

## Project Goal

The main goal of this project is to explore how **NLP and machine learning can be used to predict project approval based on textual information**.
