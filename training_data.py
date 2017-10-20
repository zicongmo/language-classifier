import random
import numpy as np

class LanguageTrainingData:
    # The names of the files containing language data for 2 languages
    def __init__(self, language_1, language_2, batch_size, 
                    num_char_types=32, max_word_length=10):
        self.data = {}
        self.test_data = {}
        self.batch_size = batch_size
        self.words_in_language_1 = 0
        self.words_in_language_2 = 0
        self.num_char_types = num_char_types
        self.max_word_length = max_word_length

        # If a word is in both files, the word is assigned language_2
        with open(language_1, encoding="ISO-8859-1") as file:
            words = file.readlines()
            i = 0
            for word in words:
                # ignore the newline character
                if word[-1] == "\n":
                    word = word[:-1]
                word = str.lower(word)
                if len(word) <= self.max_word_length:
                    if i % 10 == 0:
                        self.test_data[word] = 1
                    else:
                        self.data[word] = 1
                    self.words_in_language_1 += 1
                    i += 1

        with open(language_2, encoding="ISO-8859-1") as file:
            words = file.readlines()
            i = 0
            for word in words:
                if len(word) <= self.max_word_length:
                    if i % 10 == 0:
                        self.test_data[word] = 0
                    else:
                        self.data[word] = 0
                    self.words_in_language_2 += 1
                    i += 1

        # The order to use when generating the next batch of training examples
        self.items = list(self.data.items())
        self.randomize_dict()
        
        # Number of training examples provided
        self.num_examples = len(self.items)
        self.num_tests = len(self.test_data)

        print("Total training data: ", self.num_examples)
        print("Total test data: ", self.num_tests)

    # Converts string to one-hot
    # Puts all the special character at the end of vector to make id easier
    # á = 225   ó = 243
    # é = 233   ú = 250
    # í = 237   ñ = 241
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
            index = self.which_special(char_code)
            for i in range(6):
                if index == i:
                    arr.append(1)
                else:
                    arr.append(0)

        # Fill the rest in with 0's
        for i in range(len(arr), (self.num_char_types*self.max_word_length)):
            arr.append(0)

        return arr

    # Returns 0 for á, 1 for é, 2 for í,
    # 3 for ñ, 4 for ó, 5 for ú, -1 if not one of these
    def which_special(self, char_code):
        if char_code == 225:
            return 0
        if char_code == 233:
            return 1
        if char_code == 237:
            return 2
        if char_code == 241:
            return 3
        if char_code == 243:
            return 4
        if char_code == 250:
            return 5
        return -1

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
        batch_x = np.zeros(shape=(self.batch_size, self.max_word_length * self.num_char_types))
        batch_y = np.zeros(shape=(self.batch_size,2))
        upper_limit = self.index+n
        # Not enough training examples left to complete request, shuffle
        if upper_limit > self.num_examples:
            self.randomize_dict()
            upper_limit = n
            print("Reached end of training data, reshuffling")

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
        batch_x = np.zeros(shape=(self.num_tests, self.max_word_length * self.num_char_types))
        batch_y = np.zeros(shape=(self.num_tests, 2))
        i = 0
        for pair in self.test_data.items():
            word = pair[0]
            value = pair[1]
            batch_x[i] = self.convert_word(word)
            batch_y[i] = self.convert_label(value)
            i += 1
        return batch_x, batch_y

