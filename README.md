Genre Classification Project

Task Objectives

This project focuses on classifying movie genres using machine learning and deep learning techniques. The dataset consists of movie titles, descriptions, and corresponding genres. The main objectives are:

Preprocess Text Data: Combine title and description, remove stopwords, and apply TF-IDF vectorization.

Genre Encoding: Convert categorical genre labels into numerical representations using LabelEncoder.

Model Training:

Naïve Bayes Classifier: Train a Multinomial Naïve Bayes model using TF-IDF features.

Deep Learning Model: Implement a neural network using TensorFlow/Keras for genre classification.

Evaluation: Assess model performance using accuracy and classification reports.

Steps to Run the Project

1. Install Required Libraries

Ensure you have the necessary Python libraries installed. Run the following command:

pip install numpy pandas matplotlib seaborn scikit-learn nltk tensorflow keras

2. Load the Dataset

The dataset is loaded from a given directory and contains movie titles, descriptions, and genres. The train and test datasets are separated.

3. Preprocess the Data

Combine the title and description into a single feature.

Apply TF-IDF vectorization to convert text data into numerical representations.

Encode genres using LabelEncoder.

4. Train Machine Learning Model

Train a Multinomial Naïve Bayes Classifier.

Evaluate its accuracy on the training and test datasets.

5. Train Deep Learning Model

Define a Sequential Neural Network with dense layers.

Compile using Adam optimizer and Sparse Categorical Crossentropy loss.

Convert the sparse TF-IDF matrix to dense format before feeding it into the model.

Train the model for 10 epochs with a batch size of 5120.

6. Model Evaluation

Evaluate performance on the test set.

Generate and display a classification report.

Compare accuracy between Naïve Bayes and Deep Learning models.

7. Run the Project in Kaggle Notebook

To execute the project in a Kaggle notebook:

Upload the dataset if not already available.

Run the notebook cells sequentially.

Analyze the outputs, visualizations, and performance metrics.

Conclusion

This project demonstrates genre classification using both traditional machine learning and deep learning models. It provides insights into text processing techniques and model evaluation strategies for text-based classification tasks.

