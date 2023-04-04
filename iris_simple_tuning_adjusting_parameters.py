import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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

def create_network(optimizer, kernel_initializer, activation, neurons, dropout):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, activation = activation, input_dim = 4))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = neurons, activation = activation, 
                         kernel_initializer = kernel_initializer))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = 3, activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy',
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=create_network, epochs=100, batch_size=10)

parameters = {
    'batch_size': [10, 30],  
    'epochs': [1000, 2000],
    'optimizer': ['adam', 'sgd'],
    'dropout': [0.2, 0.3]
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'neurons': [4, 8, 16]
}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           cv = 2)

grid_search = grid_search.fit(predictors, classes)
best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_

# results = cross_val_score(estimator = classifier,
#                          X = predictors, y = classes,
#                          cv = 10, scoring = 'accuracy')
# mean = results.mean() 
# deviation = results.std()



