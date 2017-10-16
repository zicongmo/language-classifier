import random
import numpy as np

class LanguageTrainingData:
    # The names of the files containing language data for 2 languages
    def __init__(self, language_1, language_2):
        self.data = {}
        self.test_data = {}
        self.words_in_language_1 = 0
        self.words_in_language_2 = 0
        self.test_words_in_l1 = 0
        # If a word is in both files, the word is assigned language_2
        with open(language_1) as file:
            words = file.readlines()
            i = 0
            for word in words:
                # ignore the newline character
                if len(word) <= 10:
                    if i % 10 == 0:
                        self.test_data[word[:-1]] = 1
                        self.test_words_in_l1 += 1
                    else:
                        self.data[word[:-1]] = 1
                    self.words_in_language_1 += 1
                    i += 1
        with open(language_2) as file:
            words = file.readlines()
            i = 0
            for word in words:
                if len(word) <= 10:
                    if i % 10 == 0:
                        self.test_data[word[:-1]] = 0
                    else:
                        self.data[word[:-1]] = 0
                    self.words_in_language_2 += 1
                    i += 1
        # The order to use when generating the next batch of training examples
        self.items = list(self.data.items())
        self.randomize_dict()
        
        # Number of training examples provided
        self.num_examples = len(self.items)
        self.num_tests = len(self.test_data)

    # Converts string to one-hot
    def convert_word(self, string):
        arr = []
        for i in range(len(string)):
            lst = []
            char_code = ord(string[i])
            # If the character is between a-z, no other characters exist :)
            if (char_code >= 97) and (char_code <= 122):
                for letter in range(97, 123):
                    if letter == char_code:
                        lst.append(1)
                    else:
                        lst.append(0)
                arr.extend(lst)
        # Fill the rest in with 0's
        for i in range(len(arr), 260):
            arr.append(0)

        return arr

    # Converts label to one-hot
    def convert_label(self, label):
        if label == 1:
            return [0, 1]
        return [1, 0]

    # Resets index and randomizes the order of the items to iterate through
    def randomize_dict(self):
        self.index = 0
        random.shuffle(self.items)
    
    # Return the next n training examples
    def get_next_batch(self, n):
        batch_x = np.zeros(shape=(100,260))
        batch_y = np.zeros(shape=(100,2))
        upper_limit = self.index+n
        # Not enough training examples left to complete request, give remaining
        if upper_limit > self.num_examples:
            upper_limit = self.num_examples

        i = 0
        while self.index < upper_limit:
            pair = self.items[self.index]
            word = pair[0]
            value = pair[1]
            batch_x[i] = self.convert_word(word)
            batch_y[i] = self.convert_label(value)
            self.index += 1
            i += 1
            
        if self.index == self.num_examples:
            self.randomize_dict()
        
        return batch_x, batch_y

    # Return all of the test data
    def get_test_data(self):
        batch_x = np.zeros(shape=(self.num_tests, 260))
        batch_y = np.zeros(shape=(self.num_tests, 2))
        i = 0
        for pair in self.test_data.items():
            word = pair[0]
            value = pair[1]
            batch_x[i] = self.convert_word(word)
            batch_y[i] = self.convert_label(value)
            i += 1
        return batch_x, batch_y

