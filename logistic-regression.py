import tensorflow as tf
from training_data import LanguageTrainingData
import numpy as np
import random

train_model = "/tmp/log-reg.ckpt"

###### PARAMETERS AND CONSTANTS ######
learning_rate = 0.01
num_steps = 40000
batch_size = 100

char_types = 32
max_word_length = 10

num_inputs = char_types * max_word_length

###### SEEDING AND INITIALIZATION ######
random.seed(123)
np.random.seed(456)
tf.set_random_seed(789) # no idea why this doesn't work

tf.reset_default_graph()

language_files = ["./TrainingData/English.txt", "./TrainingData/Spanish.txt"]
num_outputs = len(language_files)

data = LanguageTrainingData(language_files, batch_size, 
								num_char_types=char_types, 
								max_word_length = max_word_length)
data.load_data()

##### MODEL INITIALIZATIOn #####
x = tf.placeholder(tf.float32, [None, num_inputs])
y = tf.placeholder(tf.float32, [None, num_outputs])

W = tf.Variable(tf.zeros([num_inputs, num_outputs])) # 
b = tf.Variable(tf.zeros([num_outputs]))


##### OUTPUTS #####
pred = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Saver to restore and save the model 
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(num_steps):
		batch = data.get_next_batch(batch_size)
		if i % 100 == 0:
			training_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
			training_cost = cost.eval(feed_dict={x: batch[0], y: batch[1]})
			print("Step ", i)
			print("\tAccuracy: ", training_accuracy)
			print("\tCost: ", training_cost)
		optimizer.run(feed_dict={x: batch[0], y: batch[1]})

	test_data = data.get_test_data()
	test_accuracy = accuracy.eval(feed_dict={x: test_data[0], y: test_data[1]})
	print("Test accuracy: {}".format(test_accuracy))

	saver.save(sess, train_model)
	print("Model saved in ", train_model)
