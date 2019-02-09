import random
import numpy as np

class LanguageTrainingData:
    # The names of the files containing language data for 2 languages
    def __init__(self, language_files, batch_size, 
                    num_char_types=32, max_word_length=10):
        self.train_data = {}
        self.test_data = {}
        self.valid_data = {}
        self.items = []
        self.batch_size = batch_size
        self.num_languages = len(language_files)
        self.num_char_types = num_char_types
        self.max_word_length = max_word_length
        self.language_files = language_files

    def load_data(self):
        print("Loading data...")
        for x in range(len(self.language_files)):
            with open(self.language_files[x], encoding="ISO-8859-1") as file:
                words = file.readlines()
                i = 0
                for word in words:
                    if word[-1] == "\n":
                        word = word[:-1]
                    word = str.lower(word)
                    if len(word) <= self.max_word_length:
                        category = 0
                        if i % 10 == 0:
                            category = 1
                        if i % 20 == 0:
                            category = 2

                        if category == 0:
                            self.train_data[word] = x
                        elif category == 1:
                            self.test_data[word] = x
                        else:
                            self.valid_data[word] = x
                        i += 1

        # Remove all elements in the training data that are in the test/valid data
        # https://stackoverflow.com/questions/8995611/removing-multiple-keys-from-a-dictionary-safely
        print("Removing overlap...")
        for word in list(self.valid_data.keys()):
            self.train_data.pop(word, None)
        for word in list(self.test_data.keys()):
            self.train_data.pop(word, None)

        # The order to use when generating the next batch of training examples
        self.items = list(self.train_data.items())
        self.randomize_dict()
        
        # Number of training examples provided
        self.num_examples = len(self.items)
        self.num_tests = len(self.test_data)
        self.num_valid = len(self.valid_data)

        print("Total training data: ", self.num_examples)
        print("Total test data: ", self.num_tests)
        print("Total validation data: ", self.num_valid)


    # Converts a one-hot matrix back into a string
    def convert_onehot(self, onehot):
        onehot = np.reshape(onehot, (10,32))
        # print(onehot)
        arr = []
        for letter in onehot:
            for i in range(len(letter)):
                if letter[i] == 1 and i <26:
                    arr.append(chr(97+i))
                elif letter[i] == 1:
                    # switches are overrated
                    if i == 26: # a accent
                        arr.append(chr(225))
                    elif i == 27: # e accent
                        arr.append(chr(233))
                    elif i == 28: # i accent
                        arr.append(chr(237))
                    elif i == 29: # n tilde
                        arr.append(chr(241))
                    elif i == 30: # o accent
                        arr.append(chr(243))
                    else: # u accent
                        arr.append(chr(250))

        return ''.join(arr)

    # Converts string to one-hot
    # Puts all the special character at the end of vector to make id easier
    # á = 225   ó = 243
    # é = 233   ú = 250
    # í = 237   ñ = 241
    def convert_word(self, string):
        arr = []
        for i in range(len(string)):
            char_code = ord(string[i])
            # If the character is between a-z, no other characters exist :)
            if (char_code >= 97) and (char_code <= 122):
                for letter in range(97, 123):
                    if letter == char_code:
                        arr.append(1)
                    else:
                        arr.append(0)
                for _ in range(6):
                    arr.append(0)
            else:
                for _ in range(26):
                    arr.append(0)

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
        if char_code == 225 or char_code == 224:
            return 0
        if char_code == 233 or char_code == 232:
            return 1
        if char_code == 237 or char_code == 236:
            return 2
        if char_code == 241:
            return 3
        if char_code == 243 or char_code == 242:
            return 4
        if char_code == 250 or char_code == 249:
            return 5
        return -1

    # Converts label to one-hot
    def convert_label(self, label):
        lst = [0] * self.num_languages
        for i in range(self.num_languages):
            if i == label:
                lst[i] = 1
        return lst

    # Resets index and randomizes the order of the items to iterate through
    def randomize_dict(self):
        self.index = 0
        random.shuffle(self.items)
    
    # Return the next n training examples
    def get_next_batch(self, n):
        if len(self.items) == 0:
            raise ValueError("Error: Called get_next_batch without loading data")

        batch_x = np.zeros(shape=(self.batch_size, self.max_word_length * self.num_char_types))
        batch_y = np.zeros(shape=(self.batch_size, self.num_languages))
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

    def get_train_data(self):
        if len(self.train_data) == 0:
            raise ValueError("Error: Called get_valid_data without loading data")
            
        batch_x = np.zeros(shape=(self.num_examples, self.max_word_length * self.num_char_types))
        batch_y = np.zeros(shape=(self.num_examples, self.num_languages))
        i = 0
        for pair in self.train_data.items():
            word = pair[0]
            value = pair[1]
            batch_x[i] = self.convert_word(word)
            batch_y[i] = self.convert_label(value)
            i += 1
        return batch_x, batch_y

    def get_valid_data(self):
        if len(self.valid_data) == 0:
            raise ValueError("Error: Called get_valid_data without loading data")
            
        batch_x = np.zeros(shape=(self.num_valid, self.max_word_length * self.num_char_types))
        batch_y = np.zeros(shape=(self.num_valid, self.num_languages))
        i = 0
        for pair in self.valid_data.items():
            word = pair[0]
            value = pair[1]
            batch_x[i] = self.convert_word(word)
            batch_y[i] = self.convert_label(value)
            i += 1
        return batch_x, batch_y

    # Return all of the test data
    def get_test_data(self):
        if len(self.test_data) == 0:
            raise ValueError("Error: Called get_test_data without loading data")

        batch_x = np.zeros(shape=(self.num_tests, self.max_word_length * self.num_char_types))
        batch_y = np.zeros(shape=(self.num_tests, self.num_languages))
        i = 0
        for pair in self.test_data.items():
            word = pair[0]
            value = pair[1]
            batch_x[i] = self.convert_word(word)
            batch_y[i] = self.convert_label(value)
            i += 1
        return batch_x, batch_y

