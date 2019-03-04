import tensorflow as tf
from training_data import LanguageTrainingData
import numpy as np
import random

learning_rate = 1e-3
batch_size = 100
num_steps = 20000
char_types = 32
max_word_length = 8

num_inputs = char_types * max_word_length

random.seed(123)
np.random.seed(456)
tf.set_random_seed(789)

tf.reset_default_graph()

language_files = ["./TrainingData/English.txt", "./TrainingData/Spanish.txt"]
num_outputs = len(language_files)

data = LanguageTrainingData(language_files, batch_size, 
								num_char_types=char_types, 
								max_word_length = max_word_length)
data.load_data()

x = tf.placeholder(tf.float32, [None, num_inputs])
# y = tf.placeholder(tf.float32, [None, num_inputs])

x_square = tf.reshape(x, [-1, max_word_length, char_types, 1])
# y_square = tf.reshape(y, [-1, max_word_length, char_types, 1])
print("Input shape:", x_square.shape)

def activation(x):
	return tf.nn.sigmoid(x)


###### Setting up the full graph ###### 
## Encoding weights
# Conv 1 Encode
# Input: (?, 8, 32, 1)
# Output: (?, 8, 32, 32)
num_features_1 = 32
filter_size = 3
total_var = num_features_1 * char_types/2 * max_word_length/2
shape = [filter_size, filter_size, 1, num_features_1]
W_conv1 = tf.Variable(tf.truncated_normal(shape=shape, stddev=pow(2/total_var, 1/2)), name = "Wconv1")
b_conv1 = tf.Variable(tf.constant(0.0, shape=[num_features_1]), name = "bconv1")
h_1 = activation(tf.nn.conv2d(x_square, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

print("Conv1 encode shape:", h_1.shape)

# Maxpool
h_1 = tf.nn.max_pool(h_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

print("Maxpool shape:", h_1.shape)

# Conv 2 Encode
# Input: (?, 4, 16, 32)
# Output: (?, 4, 16, 32)
num_features_2 = 32
filter_size = 3
total_var = num_features_2 * char_types/2 * max_word_length/2
shape = [filter_size, filter_size, num_features_1, num_features_2]
W_conv2 = tf.Variable(tf.truncated_normal(shape=shape, stddev=pow(2/total_var, 1/2)), name = "Wconv2")
b_conv2 = tf.Variable(tf.constant(0.0, shape=[num_features_2]), name = "bconv2")
h_2 = activation(tf.nn.conv2d(h_1, W_conv2, strides=[1,1,1,1], padding="SAME") + b_conv2)

print("Conv2 encode shape:", h_2.shape)

# FC 1 Encode
# Input: (?, 4, 16, 32)
# Output: (?, 256)
h_long = tf.reshape(h_2, [-1, 4 * 16 * 32])
total_var = 256 * 4 * 16 * 32
shape = [4 * 16 * 32, 256]
W_3 = tf.Variable(tf.truncated_normal(shape=shape, stddev=pow(2/total_var, 1/2)), name = "W3")
b_3 = tf.Variable(tf.constant(0.0, shape=[256]), name = "b3e")
h_3 = activation(tf.matmul(h_long, W_3) + b_3)

print("FC1 encode shape:", h_3.shape)

# FC 2 Encode
# Input: (?, 256)
# Output: (?, 128)
total_var = 256 * 128
shape = [256, 128]
W_4 = tf.Variable(tf.truncated_normal(shape=shape, stddev=pow(2/total_var, 1/2)), name = "W4")
b_4 = tf.Variable(tf.constant(0.0, shape=[128]), name = "b4e")
h_enc = activation(tf.matmul(h_3, W_4) + b_4)

print("FC2 encode shape:", h_enc.shape)

## Decoding weights
# FC 2 Decode
# Input: (?, 128)
# Output: (?, 256)
b_4d = tf.Variable(tf.constant(0.0, shape=[256]), name = "b4d")
h_4 = activation(tf.matmul(h_enc, tf.transpose(W_4)) + b_4d)

print("FC2 decode shape", h_4.shape)

# FC 1 Decode
# Input: (?, 256)
# Output: (?, 4, 16, 32)
b_3d = tf.Variable(tf.constant(0.0, shape=[4 * 16 * 32]), name = "b3d")
h_long = activation(tf.matmul(h_4, tf.transpose(W_3)) + b_3d)
h_3 = tf.reshape(h_long, [-1, 4, 16, 32])

print("FC1 decode shape:", h_3.shape)

# Conv 2 Decode, use a FC layer to convert into a (?,8,32,32) image for conv 1 decode
# Input: (?, 4, 16, 32)
# Output: (?, 8, 32, 32)
x_long = tf.reshape(h_2, [-1, 4 * 16 * 32])
total_var = 4 * 16 * 32
shape = [4 * 16 * 32, 8 * 32 * 32]
W_fc2 = tf.Variable(tf.truncated_normal(shape=shape, stddev=pow(2/total_var, 1/2)), name = "Wfc2")
b_fc2 = tf.Variable(tf.constant(0.0, shape=[8*32*32]), name = "bfc2")
h_2 = tf.nn.sigmoid(tf.matmul(x_long, W_fc2) + b_fc2)
h_square = tf.reshape(h_2, [-1, 8, 32, 32])

print("Conv2 decode shape:", h_square.shape)

# Conv 1 Decode
# Input: (?, 8, 32, 32)
# Output: (?, 8, 32, 1)
num_features = 1
filter_size = 3
total_var = num_features * char_types/2 * max_word_length/2
shape = [filter_size, filter_size, 32, num_features]
W_dec1 = tf.Variable(tf.truncated_normal(shape=shape, stddev=pow(2/total_var, 1/2)), name = "Wdec1")
b_dec1 = tf.Variable(tf.constant(0.0, shape=[num_features]), name = "bdec1")
logits = tf.nn.conv2d(h_square, W_dec1, strides=[1, 1, 1, 1], padding="SAME") + b_dec1

output = tf.nn.sigmoid(logits)

print("Decoded shape:", logits.shape)

# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_square, logits=logits)
# cost = tf.reduce_mean(loss)
# opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
	# sess.run(tf.global_variables_initializer())

	saver.restore(sess, "./models/auto-encoder-layer2.ckpt")

	# for i in range(num_steps):
	# 	batch = data.get_next_batch(batch_size)
	# 	if i % 100 == 0:
	# 		training_cost = cost.eval(feed_dict={x: batch[0], y: batch[0]})
	# 		print("Step ", i)
	# 		print("\tCost: ", training_cost)

	# 	opt.run(feed_dict={x: batch[0], y: batch[0]})

	test_data = data.get_test_data()
	# test_cost = cost.eval(feed_dict={x: test_data[0], y: test_data[0]})

	test_data_format = np.reshape(test_data[0], [7414, 8, 32, 1])	

	# reconstructed = output.eval(feed_dict = {x: test_data[0]})
	# for image in reconstructed:
	# 	print(image)

	for image in test_data_format:
		print(image)

	# for image in reconstructed:
		# print(image.shape)

	# saver.save(sess, "./models/auto-encoder-layer2.ckpt")
