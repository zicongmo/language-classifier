class TrainingLanguageData:
    # The names of the files containing language data for 2 languages
    def __init__(self, language_1, language_2):
        self.data = {}
        # If a word is in both files, the word is assigned language_2
        with open(language_1) as file:
            word = file.read()
            self.data[word] = 1
        with open(language_2) as file:
            word = file.read()
            self.data[word] = 0
            
data = TrainingLanguageData("English.txt", "Spanish.txt")
