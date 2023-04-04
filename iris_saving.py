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

new_register = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

register_predict = classifier.predict(new_register)
register_predict_bin = (register_predict > 0.5)

# Saving
classifier_json = classifier.to_json()
with open('classifier_iris.json', 'w') as json_file:
    json_file.write(classifier_json)
classifier.save_weights('classifier_iris.h5')
