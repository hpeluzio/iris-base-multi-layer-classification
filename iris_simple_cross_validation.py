import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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

def create_network():
    classifier = Sequential()
    classifier.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classifier.add(Dense(units = 4, activation = 'relu'))
    classifier.add(Dense(units = 3, activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                       metrics = ['categorical_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = create_network, 
                             epochs = 1000, 
                             batch_size = 10)
results = cross_val_score(estimator = classifier,
                          X = predictors, y = classes,
                          cv = 10, scoring = 'accuracy')
mean = results.mean() 
deviation = results.std()



