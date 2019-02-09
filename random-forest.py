import numpy as np
from sklearn.ensemble import RandomForestClassifier
from training_data import LanguageTrainingData
import random

batch_size = 100

char_types = 32
max_word_length = 10

num_inputs = char_types * max_word_length

random.seed(123)
np.random.seed(456)

language_files = ["./TrainingData/English.txt", "./TrainingData/Spanish.txt"]
num_outputs = len(language_files)

data = LanguageTrainingData(language_files, batch_size, 
								num_char_types=char_types, 
								max_word_length = max_word_length)
data.load_data()

train_x, train_y = data.get_train_data()
test_x, test_y = data.get_test_data()

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(train_x, train_y)

pred = rf.predict(test_x)
count = 0
for i in range(len(pred)):
	if pred[i][0] == test_y[i][0]:
		count += 1

print(count/len(test_y))
