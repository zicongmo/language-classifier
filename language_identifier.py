import tensorflow as tf
from training_data import LanguageTrainingData

data = LanguageTrainingData("English.txt", "Spanish.txt")

# classify between English and Spanish
num_languages = 2

# Number of nodes in hidden layers 1 and 2
num_nodes_l1 = 500
num_nodes_l2 = 500

# Batch size
batch_size = 100

# Number of training epochs
num_epochs = 1000

# Dimension of input vector
num_features = 260

x = tf.placeholder("float", shape=[None, num_features])
y = tf.placeholder("float", shape=[None, num_languages])

# input_data is a num_features dimensional vector
def neural_network(input_data):
    # Initialize each of our layers to have initially random weights and biases
    layer1 = {"weights": tf.Variable(tf.random_normal([num_features, num_nodes_l1])),
               "biases": tf.Variable(tf.random_normal([num_nodes_l1]))}
    layer2 = {"weights": tf.Variable(tf.random_normal([num_nodes_l1, num_nodes_l2])),
               "biases": tf.Variable(tf.random_normal([num_nodes_l2]))}
    output_layer = {"weights": tf.Variable(tf.random_normal([num_nodes_l2, num_languages])),
                     "biases": tf.Variable(tf.random_normal([num_languages]))}

    # Feed the input of each layer through the layer until the output layer
    output_l1 = tf.add(tf.matmul(input_data, layer1["weights"]) , layer1["biases"])
    output_l1 = tf.nn.relu(output_l1)
                                                                  
    output_l2 = tf.add(tf.matmul(output_l1, layer2["weights"]) , layer2["biases"])
    output_l2 = tf.nn.relu(output_l2)

    output = tf.add(tf.matmul(output_l2, output_layer["weights"]) , output_layer["biases"])
    
    return output

# input_data is a list containing all of the training data
def train(x):
    prediction = neural_network(x)
    # Computes the cross entropy between prediction and y, and takes the average
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Change the NN variables with the Adam algorithm to minimize the value of cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(int(data.num_examples/batch_size)):
                batch_x, batch_y = data.get_next_batch(batch_size)
                i, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                total_loss += c
            if epoch % 100 == 0:
                print("Epoch", epoch, "completed with loss:", total_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        batch_x, batch_y = data.get_test_data()
        print('Accuracy:', accuracy.eval({x: batch_x, y: batch_y}))
            
train(x)
        
