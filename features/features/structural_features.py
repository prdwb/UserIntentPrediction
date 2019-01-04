from data_helper import tokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def post_length(utterances):
    length = []
    unique_length = []
    unique_stemmed_length = []

    stop = set(stopwords.words('english'))
    ps = PorterStemmer()

    for utterance in utterances:
        tokens = tokenizer(utterance)
        tokens = [token for token in tokens if token not in stop]

        unique_tokens = list(set(tokens))
        unique_tokens_stemmed = list(set([ps.stem(token) for token in unique_tokens]))

        length.append(len(tokens))
        unique_length.append(len(unique_tokens))
        unique_stemmed_length.append(len(unique_tokens_stemmed))

    return length, unique_length, unique_stemmed_length


if __name__ == '__main__':
    utterances = ['play, played, plays, played', 'Hi!, my name is Aaron. What is yours']
    length, unique_length, unique_stemmed_length = post_length(utterances)
    print(length)
    print(unique_length)
    print(unique_stemmed_length)