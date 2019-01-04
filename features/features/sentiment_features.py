from nltk.sentiment.vader import SentimentIntensityAnalyzer
from data_helper import tokenizer, load_sentiment_lexicon


def sentiment_scores(sentences):
    res = []
    sid = SentimentIntensityAnalyzer()
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        res.append(list(map(str, [ss['neg'], ss['neu'], ss['pos']])))
    return res


def thank(utterances):
    res = []
    for utterance in utterances:
        if utterance.find('thank') != -1:
            res.append(1)
        else:
            res.append(0)
    return res


def exclamation_mark(utterances):
    res = []
    for utterance in utterances:
        if utterance.find('!') != -1:
            res.append(1)
        else:
            res.append(0)
    return res


def ve_feedback(utterances):
    res = []
    for utterance in utterances:
        if utterance.find('did not') != -1 or utterance.find('does not') != -1:
            res.append(1)
        else:
            res.append(0)
    return res


def lexicon(utterances, pos_dict, neg_dict):
    res = []
    for utterance in utterances:
        pos_count, neg_count = 0, 0
        tokens = tokenizer(utterance)
        for token in tokens:
            if token in pos_dict:
                pos_count += 1
            elif token in neg_dict:
                neg_count += 1
        res.append([str(pos_count), str(neg_count)])
    return res


if __name__ == '__main__':
    utterances = ['bad movie', 'thank you for your help, but the solution does not work']
    pos_file = '../data/positive-words.txt'
    neg_file = '../data/negative-words.txt'
    pos_dict, neg_dict = load_sentiment_lexicon(pos_file, neg_file)
    print(lexicon(utterances, pos_dict, neg_dict))