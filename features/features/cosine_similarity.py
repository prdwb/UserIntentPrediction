import math
from collections import Counter
from data_helper import tokenizer
from pprint import pprint
from copy import deepcopy

def compute_tf_idf_cosine(q1, q2, term_to_idf_dict):
    q1_term_count = Counter(q1)
    q2_term_count = Counter(q2)
    idf = 10.8983491402  # default value
    for t in q1_term_count.keys():
        idf = term_to_idf_dict[t]
        q1_term_count[t] *= idf

    for t in q2_term_count.keys():
        idf = term_to_idf_dict[t]
        q2_term_count[t] *= idf

    cosine = 0.0
    q1_l2_norm = 0.0
    q2_l2_norm = 0.0
    for q1_term in q1_term_count.keys():
        q1_l2_norm += math.pow(q1_term_count[q1_term],2)
        if q1_term not in q2_term_count.keys():
            continue
        else:
            cosine += q1_term_count[q1_term] * q2_term_count[q1_term]

    for q2_term in q2_term_count.keys():
        q2_l2_norm += math.pow(q2_term_count[q2_term],2)
    #normalize
    # print('q1, q2: ', q1, q2)
    try:
        return cosine / (math.sqrt(q1_l2_norm) * math.sqrt(q2_l2_norm))
    except:
        return 0

def cosine_similarity(title, utterances, term_to_idf_dict):
    title_sim, init_sim, thread_sim = [], [], []

    title_tokens = list(filter(lambda x: x in term_to_idf_dict, tokenizer(title)))
    # print(title_tokens)
    utterances_tokens = []

    for i, utterance in enumerate(utterances):
        utterance_tokens = list(filter(lambda x: x in term_to_idf_dict, tokenizer(utterance)))
        utterances_tokens.append(utterance_tokens)

    # pprint(utterances_tokens)

    init_post = deepcopy(utterances_tokens[0])
    thread_tokens = deepcopy(title_tokens)
    for utterance_tokens in utterances_tokens:
        thread_tokens += utterance_tokens

    for utterance_tokens in utterances_tokens:
        title_sim.append(compute_tf_idf_cosine(title_tokens, utterance_tokens, term_to_idf_dict))
    for utterance_tokens in utterances_tokens:
        init_sim.append(compute_tf_idf_cosine(init_post, utterance_tokens, term_to_idf_dict))
    for utterance_tokens in utterances_tokens:
        thread_sim.append(compute_tf_idf_cosine(thread_tokens, utterance_tokens, term_to_idf_dict))

    return title_sim, init_sim, thread_sim

