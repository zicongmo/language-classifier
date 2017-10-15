import random

class TrainingLanguageData:
    # The names of the files containing language data for 2 languages
    def __init__(self, language_1, language_2):
        self.data = {}
        # If a word is in both files, the word is assigned language_2
        with open(language_1) as file:
            words = file.readlines()
            for word in words:
                # ignore the newline character
                self.data[word[:-1]] = 1    
        with open(language_2) as file:
            words = file.readlines()
            for word in words:
                self.data[word[:-1]] = 0
        # The order to use when generating the next batch of training examples
        self.items = list(self.data.items())
        self.randomize_dict()
        
        # Number of training examples provided
        self.num_examples = len(self.items)

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
        return arr

    # Resets index and randomizes the order of the items to iterate through
    def randomize_dict(self):
        self.index = 0
        random.shuffle(self.items)
    
    # Return the next n training examples as a list of lists
    def get_next_batch(self, n):
        batch = []
        upper_limit = self.index+n
        # Not enough training examples left to complete request, give remaining
        if upper_limit > self.num_examples:
            upper_limit = self.num_examples
            
        while self.index < upper_limit:
            pair = self.items[self.index]
            word = pair[0]
            value = pair[1]
            lst = [self.convert_word(word), value]
            batch.append(lst)
            self.index += 1
            
        if self.index == self.num_examples:
            self.randomize_dict()
            
        return batch
