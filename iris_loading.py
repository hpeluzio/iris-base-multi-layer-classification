import numpy as np
import pandas as pd
from keras.models import model_from_json

file = open('classifier_iris.json', 'r')
network_structure = file.read()
file.close()

classifier = model_from_json(network_structure)
classifier.load_weights('classifier_iris.h5')

# Classifying one register
new_register = np.array([[4.9, 3, 1.4, 0.2]])

register_predict = classifier.predict(new_register)
register_predict_bin = (register_predict > 0.5)