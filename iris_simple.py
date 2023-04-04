import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

base = pd.read_csv('iris.csv')
predictors = base.iloc[:,0:4].values
classes = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classes = labelencoder.fit_transform(classes)
classes_dummy = np_utils.to_categorical(classes)
# iris setosa      1 0 0
# iris virginica   0 1 0
# iris versicolor  0 0 1

from sklearn.model_selection import train_test_split
predictors_training, predictors_test, classes_training, classes_test = train_test_split(
    predictors, classes_dummy, test_size=0.25)

classifier = Sequential()
classifier.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classifier.add(Dense(units = 4, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                   metrics = ['categorical_accuracy'])
classifier.fit(predictors_training, classes_training, batch_size = 10,
               epochs = 1000)
result = classifier.evaluate(predictors_test, classes_test)
predictions = classifier.predict(predictors_test)
predictions = (predictions > 0.5)
import numpy as np
classes_test2 = [np.argmax(t) for t in classes_test]
predictions2 = [np.argmax(t) for t in predictions]

from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(predictions2, classes_test2)