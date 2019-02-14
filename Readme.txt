This project is implemented solely for learning purposes.
The code and project direction is influenced by https://medium.com/@martinpella/customers-tweets-classification-41cdca4e2de

The dataset used for this project is called "Airline sentiment tweets" and can be found here: https://www.kaggle.com/tango911/airline-sentiment-tweets


In this project, first pre-processing steps are taken place. In this text cleaning and any unwanted noise is removed.
The data are split into training and test set for better learning of the model.
Text vectorization is implemented after this using the Tf-IFD method and sparse vectors are generated.
Logistic regression is used to train the model.


The model is able to achieve an average of 80% accuracy. It is able to classify tweets into neutral, negative and positive sentiments with this accuracy. 
