from sklearn.feature_extraction.text import CountVectorizer
from data_helper import *

def generate_vocab(data):

    # count_vect = CountVectorizer(stop_words='english')
    count_vect = CountVectorizer(tokenizer=tokenizer, stop_words='english'  )
    counts = count_vect.fit_transform(data)

    return count_vect.vocabulary_

if __name__ == '__main__':
    conn_title = connect_db()
    conn_utter = connect_db()

    sql_title = 'select title from titles_final'
    sql_utter = 'select utterance from contents_final'

    with conn_title.cursor() as cursor_title, conn_utter.cursor() as cursor_utter:
        cursor_title.execute(sql_title)
        titles = [row['title'] for row in cursor_title.fetchall()]

        cursor_utter.execute(sql_utter)
        utterances = [row['utterance'] for row in cursor_utter.fetchall()]

    vocab_file = 'data/vocab.tsv'

    vocab = generate_vocab(map(clean_str, titles + utterances))

    with open(vocab_file, 'w') as vocab_output:
        for term in vocab:
            vocab_output.write('{0}\t{1}\n'.format(term, vocab[term]))