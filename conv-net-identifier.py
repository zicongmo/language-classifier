import tensorflow as tf
from training_data import LanguageTrainingData

learning_rate = 5e-4
num_steps = 20000
batch_size = 100

char_types = 32
max_word_length = 10

num_inputs = char_types * max_word_length
num_outputs = 2

dropout_prob = 0.5

data = LanguageTrainingData("English.txt", "Spanish.txt", batch_size, 
								num_char_types=char_types, 
								max_word_length = max_word_length)

def weight_variables(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variables(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

x = tf.placeholder(tf.float32, [None, num_inputs])
y = tf.placeholder(tf.float32, [None, num_outputs])
keep_prob = tf.placeholder(tf.float32)

# Layer 1 has 32 output features with 5x5 filter
W_conv1 = weight_variables([5, 5, 1, 32])
b_conv1 = bias_variables([32])
x_square = tf.reshape(x, [-1, max_word_length, char_types, 1])

h_c1 = tf.nn.relu(conv2d(x_square, W_conv1) + b_conv1)
h_pool1 = max_pool(h_c1)

# Layer 2 with 64 output features with 5x5 filter
W_conv2 = weight_variables([5, 5, 32, 64])
b_conv1 = bias_variables([64])

h_c2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv1)

# max/2 x char_types/2 x 64
new_length = int(max_word_length/2)
new_width = int(char_types/2)

# FC 512 neurons
W_fc1 = weight_variables([new_length*new_width*64, 512])
b_fc1 = bias_variables([512])
x_flat = tf.reshape(h_c2, [-1, new_length*new_width*64])
h_fc = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

# Dropout
h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

# Readout layer
W_output = weight_variables([512, 2])
b_output = bias_variables([2])
output = tf.matmul(h_fc_drop, W_output) + b_output

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("Learning rate: ", learning_rate)
	for i in range(num_steps):
		batch = data.get_next_batch(batch_size)
		if i % 100 == 0:
			training_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
			training_cost = cost.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
			print("Step ", i)
			print("\tAccuracy: ", training_accuracy)
			print("\tCost: ", training_cost)
		training_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: dropout_prob})

	test_data = data.get_test_data()
	test_accuracy = accuracy.eval(feed_dict={x: test_data[0], y: test_data[1], keep_prob: 1.0})
	print("Test accuracy: {0}".format(test_accuracy))
