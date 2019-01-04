from data_helper import *
from nltk.corpus import stopwords
import math

def compute_idf(sents_ints, term_to_id_dict, id_to_term_dict):
    term_to_idf_dict = dict()
    term_to_df_dict = dict()
    term_to_tf_dict = dict() # return term_to_tf_dict to compute the background probability in QL
    total_doc_number = 0.0
    stop = set(stopwords.words('english')) # remove stop words

    for sent_ints in sents_ints:
        sent = [id_to_term_dict[i] for i in sent_ints if id_to_term_dict[i] not in stop]
        update_df_dict(sent, term_to_df_dict, id_to_term_dict, term_to_tf_dict)
        total_doc_number += 1.0

    for term in term_to_df_dict:
        idf = math.log((total_doc_number - term_to_df_dict[term] + 0.5) / (term_to_df_dict[term] + 0.5))
        term_to_idf_dict[term] = idf

    # normlize term_to_tf_dict
    total_token_num = float(sum(term_to_tf_dict.values()))
    for t in term_to_tf_dict.keys():
        term_to_tf_dict[t] /= total_token_num

    return term_to_idf_dict, term_to_tf_dict

def update_df_dict(q1, term_to_df_dict, id_to_term_dict, term_to_tf_dict):
    word_set = set(q1)
    for w in q1:
        if w in term_to_tf_dict:
            term_to_tf_dict[w] += 1.0
        else:
            term_to_tf_dict[w] = 1.0
        word_set.add(w)
    for w in word_set:
        if w in term_to_df_dict:
            term_to_df_dict[w] += 1.0
        else:
            term_to_df_dict[w] = 1.0

if __name__ == '__main__':

    vocab_file = 'data/vocab.tsv'
    idf_file = 'data/idf.tsv'

    term_to_id_dict = dict()
    id_to_term_dict = dict()
    init_term_to_id_dict(term_to_id_dict, vocab_file)
    id_to_term_dict = dict(zip(term_to_id_dict.values(), term_to_id_dict.keys()))

    conn_title = connect_db()
    conn_utter = connect_db()

    sql_title = 'select title from titles_final'
    sql_utter = 'select utterance from contents_final'

    with conn_title.cursor() as cursor_title, conn_utter.cursor() as cursor_utter:
        cursor_title.execute(sql_title)
        titles = [row['title'] for row in cursor_title.fetchall()]

        cursor_utter.execute(sql_utter)
        utterances = [row['utterance'] for row in cursor_utter.fetchall()]

    sents_ints = []
    sents = map(clean_str, titles + utterances)
    for sent in sents:
        sent_ints = [term_to_id_dict.get(term) for term in tokenizer(sent)]
        sent_ints = list(filter(lambda x: x is not None, sent_ints))
        sents_ints.append(sent_ints)

    term_to_idf_dict, term_to_tf_dict = compute_idf(sents_ints, term_to_id_dict, id_to_term_dict)

    with open(idf_file, 'w') as f_out:
        for term in term_to_idf_dict:
            f_out.write(term + '\t' + str(term_to_idf_dict[term]) + '\n')
