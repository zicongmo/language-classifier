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
        self.index = 0
        random.shuffle(self.items)

        # Number of training examples provided
        self.num_examples = len(self.items)

    # Converts string to one-hot
    def convert_word(self, string):
        return string

    # Return the next n training examples as a list of lists
    def get_next_batch(self, n):
        batch = []
        upper_limit = self.index+n
        # Not enough training examples left to complete request, give remaining
        if upper_limit > self.num_examples:
            upper_limit = self.num_examples
        for i in range(self.index, upper_limit):
            word = self.items[i][0]
            value = self.items[i][1]
            lst = [self.convert_word(word), value]
            batch.append(lst)
            self.index += 1
        return batch
        
data = TrainingLanguageData("English.txt", "Spanish.txt")
print(data.get_next_batch(10))
