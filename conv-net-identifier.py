import tensorflow as tf
from training_data import LanguageTrainingData
import numpy as np
import random

gamma = 1000
test = True
test_model = "./models/model_full_sigmoid_1000.ckpt"
train_model = "./models/model_full.ckpt"

if test:
	print("TESTING: %s" % test_model )
else:
	print("TRAINING: %s" % train_model)


###### PARAMETERS AND CONSTANTS ######
learning_rate = 5e-4
num_steps = 30000
batch_size = 100

char_types = 32
max_word_length = 10

num_inputs = char_types * max_word_length

dropout_prob = 0.5


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

###### MODEL INITIALIZATION FUNCTIONS ######
def weight_variables(shape):
	n = 1
	for i in range(len(shape)):
		n *= shape[i]
	return tf.Variable(tf.truncated_normal(shape=shape, stddev=pow(2/n, 1/2)))

def bias_variables(shape):
	return tf.Variable(tf.constant(0.0, shape=shape))

def conv2d(x, W, stride=1):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool(x, k=2, stride=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding="SAME")

# 1 if x > 0, 0 otherwise
def step(x):
	return tf.cast(tf.greater(x, 0), tf.float32)

# Set all activation functions at the same time
def activation(x):
	if test:
		return step(x)
	# return tf.sigmoid(gamma*x)
	return tf.nn.relu(x)

###### HELPER FUNCTIONS ######
def log(step, train_acc, valid_acc, cost):
	training_log.append([step, train_acc, valid_acc, cost])

def log_predictions(x, y_pred, label, f="./predictions.txt"):
	with open(f, "w+") as file:
		file.write("Word\t\tPred\t\tTrue\n")
		for i in range(len(x)):
			# convert x back into character form
			test_word = data.convert_onehot(x[i]);
			file.write("{}\t\t{}\t\t{}\n".format(test_word, "Spanish" if y_pred[i] else "English", "English" if label[i][0] else "Spanish"))
		file.write("\n\n")


###### MODEL ######
x = tf.placeholder(tf.float32, [None, num_inputs])
y = tf.placeholder(tf.float32, [None, num_outputs])
keep_prob = tf.placeholder(tf.float32)

# Layer 1 has 32 output features with 3x3 filter
num_features = 32
W_conv1 = weight_variables([3, 3, 1, num_features])
b_conv1 = bias_variables([num_features])
x_square = tf.reshape(x, [-1, max_word_length, char_types, 1])

h_c1 = activation(conv2d(x_square, W_conv1) + b_conv1)
h_pool1 = max_pool(h_c1)
# h_pool2 = max_pool(h_pool1)

# Layer 2 with 32 output features with 3x3 filter
W_conv2 = weight_variables([3, 3, 32, 32])
b_conv2 = bias_variables([32])

h_c2 = activation(conv2d(h_pool1, W_conv2) + b_conv2)

# max/2 x char_types/2 x 32
new_length = int(max_word_length/2)
new_width = int(char_types/2)


# FC 256 neurons
W_fc1 = weight_variables([new_length*new_width*num_features, 256])
b_fc1 = bias_variables([256])
x_flat = tf.reshape(h_pool1, [-1, new_length*new_width*num_features])
h_fc1 = activation(tf.matmul(x_flat, W_fc1) + b_fc1)

# Dropout
h_fc_drop = tf.nn.dropout(h_fc1, keep_prob)

# FC 128 neurons
W_fc2 = weight_variables([256,128])
b_fc2 = bias_variables([128])
h_fc2 = activation(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

# Readout layer
W_output = weight_variables([128, num_outputs])
b_output = bias_variables([num_outputs])
output = tf.matmul(h_fc2, W_output) + b_output


###### OUTPUTS ######
prediction = tf.argmax(output, 1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Saver to restore and save the model 
saver = tf.train.Saver(max_to_keep = None) # keep all the models

###### EVALUATION ######
with tf.Session() as sess:
	if test:
		saver.restore(sess, test_model)
	else:
		sess.run(tf.global_variables_initializer())
	print("Learning rate: ", learning_rate)

	if not test:
		for i in range(num_steps):
			batch = data.get_next_batch(batch_size)
			if i % 100 == 0:
				training_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
				training_cost = cost.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
				print("Step ", i)
				print("\tTraining Accuracy: ", training_accuracy)
				print("\tCost: ", training_cost)

			if i % 2000 == 0:
				valid_batch = data.get_valid_data()
				valid_accuracy = accuracy.eval(feed_dict={x: valid_batch[0], y: valid_batch[1], keep_prob: 1.0})
				print("\tValidation Accuracy: ", valid_accuracy)
				
			training_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

	test_data = data.get_test_data()
	test_accuracy = accuracy.eval(feed_dict={x: test_data[0], y: test_data[1], keep_prob: 1.0})
	print("Test accuracy: {0}".format(test_accuracy))

	#  log_predictions(test_data[0], prediction.eval(feed_dict={x: test_data[0], y:test_data[1], keep_prob:1.0}), test_data[1], f="./test_predictions.txt")

	if not test:
		print("Saving model")
		save_path = saver.save(sess, train_model)
		print("Model saved in path: {}".format(save_path))

