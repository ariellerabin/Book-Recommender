
from collections import defaultdict, Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import nltk
import string
class BookRecs:

    @staticmethod
    def parser_stopwords(db, excerpt_column):
        stop_words = set(stopwords.words('english'))
        for index, row in db.iterrows():
            text = row[excerpt_column]
            # Use RegexpTokenizer to tokenize contractions
            tokenizer = RegexpTokenizer(r"\w+['\w*]*")

            # Tokenize into words
            words = tokenizer.tokenize(text.lower())

            # Print words that are not in the set of stop words
            filtered_words = [word for word in words if word not in stop_words]

            # Join the list of words into a single string
            concat = ' '.join(filtered_words)

            row[excerpt_column] = concat

        return db
    @staticmethod
    def parser(db, excerpt_column):
        for index, row in db.iterrows():
            text = row[excerpt_column]

            # Tokenize into words and punctuation
            tokens = wordpunct_tokenize(text.lower())

            # Join the list of tokens into a single string with spaces
            concat = ''
            for i, token in enumerate(tokens):
                # Add space only if the token is a word and it's not the first token
                if token.isalnum() and i > 0 and not token.startswith("'") and tokens[i - 1][-1].isalpha():
                    concat += ' '
                    concat += token
                elif token == "," or token == "." or token == ":" or token == "?" or token == "!" or token ==":":
                    concat += token
                    concat += ' '
                else:
                    concat += token

            row[excerpt_column] = concat

        return db

    def avg_sentence_length(self, df, excerpt_column, title_column):
        # keep in stop words
        """ calculates the average amount of words per sentence, including stop words """
        dict_ASL = {}
        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            lengths = []
            text = row[excerpt_column]

        # Tokenize the text into sentences
            sentences = nltk.sent_tokenize(text)


        # Get the length of each sentence
            lengths = [len(sentence.split()) for sentence in sentences]

            dict_ASL[row[title_column]] = sum(lengths)/len(lengths)
        return(dict_ASL)


    def avg_word_length(self, df, excerpt_column, title_column):
        # remove stop words
        """ calculates the average word length, omitting stop words """
        dict_AWL ={}

        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():

            sum = 0
            words_list = []
            sample = row[excerpt_column]
            words_list = sample.split()
            for words in words_list:
                sum += len(words)
            dict_AWL[row[title_column]] = (sum/len(words_list))

        return(dict_AWL)



    def punctuation_count(self, df, excerpt_column, title_column):
        # keep in stop words
        """ calculates the amount of punctuation used """
        dict_PC = {}

        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            text = row[excerpt_column]

            punctuation_count = 0
            for char in text:
                if char in string.punctuation:
                    punctuation_count += 1
            dict_PC[row[title_column]] = (punctuation_count)

        return(dict_PC)


    def avg_word_frequency(self, df, excerpt_column, title_column):
        # remove stop words
        """ calulcates the average word frequency. The average amount of times a word is used
         across all words, omitting stop words"""
        dict_AWF = {}
        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            text = row[excerpt_column]
            words_list= text.split()
            dit = dict(Counter(words_list))
            tot = 0
            for k, v in dit.items():
                tot += v
            dict_AWF[row[title_column]] = tot/len(dit)

        return(dict_AWF)



    @staticmethod
    def syllable_count(word):
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
                if word.endswith("e"):
                    count -= 1
        if count == 0:
            count += 1
        return count

    def avg_syllable_count(self, df, excerpt_column, title_column):
        # remove stop words
        """ calculates the average amount of syllables per words, omitting stop words """
        dict_ASC = {}

        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            sample = row[excerpt_column]
            words_list = sample.split()
            ASC = 0

            # Iterate over words in the list
            for word in words_list:
                ASC += BookRecs.syllable_count(word)
            dict_ASC[row[title_column]] = ASC/len(words_list)
        return(dict_ASC)

    @staticmethod
    def total_words(df, excerpt_column, title_column):
        # keep stop words
        """ for calculations of readability scores, including stop words """
        dict_TW = {}
        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            sample = row[excerpt_column]
            words_list = sample.split()

            dict_TW[row[title_column]] = len(words_list)

        return dict_TW


    @staticmethod
    def total_sentences(df, excerpt_column, title_column):
        # keep stop words
        """ for calculations of readability scores, including stop words """
        dict_TSen = {}

        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            lengths = []
            text = row[excerpt_column]

            # Tokenize the text into sentences
            sentences = nltk.sent_tokenize(text)

            # Get the length of each sentence
            count = 0
            for sentence in sentences:
                count += 1
            dict_TSen[row[title_column]] = count

        return(dict_TSen)


    @staticmethod
    def total_syllables( df, excerpt_column, title_column):
        # keep stop words
        """ for calculations of readability scores, including stop words """
        dict_TSyl = {}
        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            tot = 0
            sample = row[excerpt_column]
            words_list = sample.split()

            # Iterate over words in the list
            for word in words_list:
                tot += BookRecs.syllable_count(word)
            dict_TSyl[row[title_column]] = tot
        return(dict_TSyl)
    @staticmethod
    def total_characters(df, excerpt_column, title_column):
        # keep stop words
        """ for calculations of readability scores, including stop words """

        dict_TC = {}

        for index, row in BookRecs.parser(df.copy(), excerpt_column).iterrows():
            tot = 0
            text = row[excerpt_column]
            for l in text:
                tot += 1
            dict_TC[row[title_column]] = (tot)
        return dict_TC



    def flesch_kincaid_score(self, df, excerpt_column, title_column):
        """ calculates flesch-kincaid score per text """
        # total num words
        # total num sentences
        # total num syllables
        #206.835 − 1.015 × ( Total Words / Total Sentences ) − 84.6 × ( Total Syllables / Total Words )
        db = df.copy()
        db = BookRecs.parser(db, excerpt_column)
        dict_FKS = {}
        num_words = BookRecs.total_words(db, excerpt_column, title_column)
        num_sen = BookRecs.total_sentences(db, excerpt_column, title_column)
        num_syl = BookRecs.total_syllables(db, excerpt_column, title_column)

        # Combine dictionaries into an iterable of tuples
        combined_dicts = zip(num_words.items(), num_sen.items(), num_syl.items())
        # Iterate through the combined iterable
        for (key, words), (_, sen), (_, syl) in combined_dicts:

            dict_FKS[key] = 206.835 - (1.015*(words/sen)) - (84.6*(syl/words))

            # checked vs flesch kincaid calculator online and results are very similar :)

        return dict_FKS

    def ARI_score(self, df, excerpt_column, title_column):
        """ calculates the Automated Readability Index Per Text"""
        # 4.71 x (characters/words) + 0.5 x (words/sentences) – 21.43.
        # total num characters
        # total num words
        # total num sentences
        db = df.copy()
        db = BookRecs.parser(db, excerpt_column)
        dict_ARI = {}
        num_char = BookRecs.total_characters(db, excerpt_column, title_column)
        num_words = BookRecs.total_words(db, excerpt_column, title_column)
        num_sen = BookRecs.total_sentences(db, excerpt_column, title_column)
        combined_dicts = zip(num_char.items(), num_words.items(), num_sen.items())
        # Iterate through the combined iterable
        for (key, char), (_, words), (_, sen) in combined_dicts:
            dict_ARI[key] = ((4.71*(char/words)) + (0.5*(words/sen))) - 21.43

        return(dict_ARI)





