def question_mark(utterances):
    res = []
    for utterance in utterances:
        if utterance.find('?') != -1:
            res.append(1)
        else:
            res.append(0)
    return res


def duplicate(utterances):
    res = []
    for utterance in utterances:
        if utterance.find('same') != -1 or utterance.find('similar') != -1:
            res.append(1)
        else:
            res.append(0)
    return res


def W5H1(utterances):
    # how, what, why, who, where, when
    res = []
    for utterance in utterances:
        wh_vector = [0] * 6

        wh_vector[0] = 1 if utterance.find('how') != -1 else 0
        wh_vector[1] = 1 if utterance.find('what') != -1 else 0
        wh_vector[2] = 1 if utterance.find('why') != -1 else 0
        wh_vector[3] = 1 if utterance.find('who') != -1 else 0
        wh_vector[4] = 1 if utterance.find('where') != -1 else 0
        wh_vector[5] = 1 if utterance.find('when') != -1 else 0


        res.append(list(map(str, wh_vector)))
    return res

if __name__ == '__main__':
    utterances = ['Hi!, my what name is Aaron. What is same yours?', 'Hi!, my name is Aaron. What is yours']
    print(W5H1(utterances))