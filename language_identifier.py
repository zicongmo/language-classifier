import tensorflow as tf

# classify between English and Spanish
num_languages = 2

# Number of nodes in hidden layers 1 and 2
num_nodes_l1 = 500
num_nodes_l2 = 500

# Batch size
batch_size = 100

# Number of training epochs
num_epochs = 20

# Dimension of input vector
num_features = 100


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
    output_l1 = tf.matmul(input_data, layer1["weights"]) + layer1["biases"]
    output_l1 = tf.nn.tanh(output_l1)
                                                                  
    output_l2 = tf.matmul(output_l1, layer2["weights"]) + layer2["biases"]
    output_l2 = tf.nn.tanh(output_l2)

    output = tf.matmul(output_l2, output_layer["weights"]) + output_layer["biases"]
    output = tf.nn.tanh(output)

    return output

# input_data is a list containing all of the training data
def train(input_data):
    output = neural_network(input_data)
    # Computes the cross entropy between prediction and y, and takes the average
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Change the NN variables with the Adam algorithm to minimize the value of cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)

                          
                          

    



        
