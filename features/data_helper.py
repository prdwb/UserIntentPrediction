import re
import pymysql.cursors


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def tokenizer(doc):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    return token_pattern.findall(doc)

def init_term_to_id_dict(term_to_id_dict, vocab_file):
    with open(vocab_file, encoding='utf-8') as fin:
        for line in fin:
            #print 'line: ', line
            tokens = line.split('\t')
            term_to_id_dict[tokens[0]] = int(tokens[1].strip('\n'))


def init_tf_idf_dict(idf_file):
    term_to_idf_dict = {}

    with open(idf_file, encoding='utf-8') as fin:
        for line in fin:
            tokens = line.split('\t')
            term_to_idf_dict[tokens[0]] = float(tokens[1].strip('\n'))

    return term_to_idf_dict


def load_feature_file(target_label, f):
    # target_label e.g.: nf, pf
    # f: train, valid, test
    feature_file = 'data/{}/{}/features_all.txt'.format(target_label, f)
    data = []
    target = []
    qids = []
    with open(feature_file, encoding='utf-8') as fin:
        for line in fin:
            if line != '\n':
                tokens = line.strip().split()
                target.append(tokens[0])

                features = tokens[2:-2]
                data.append([float(feature.split(':')[1]) for feature in features])

                qid = tokens[1].split(':')[1]
                qids.append(qid)

    return data, target, qids

def load_all_utterances(target_label):
    all = 'data/{}/all_utterances.txt'.format(target_label)
    with open(all) as fin:
        x = fin.readlines()
        x = [line.strip() for line in x]
    return x

def load_sentiment_lexicon(pos_file, neg_file):


    pos_dict, neg_dict = {}, {}

    with open(pos_file) as fin:
        for line in fin:
            if line != '\n':
                line = line.strip()
                pos_dict[line] = 1

    with open(neg_file) as fin:
        for line in fin:
            if line != '\n':
                line = line.strip()
                neg_dict[line] = 1

    return pos_dict, neg_dict


def load_cnn_results(cnn_dict_file):
    cnn_results = {}
    with open(cnn_dict_file) as cnn_results_in:
        cnn_results_lines = cnn_results_in.readlines()
        for line in cnn_results_lines:
            tokens = line.strip().split('\t')
            # if int(tokens[0]) in cnn_results:
            #     print(int(tokens[0]))
            cnn_results[int(tokens[0])] = [int(float(tokens[1])), float(tokens[2]), float(tokens[3]), float(tokens[4])]
    return cnn_results

if __name__ == '__main__':
    cnn_dict_file = 'data/cnn/cnn_results/cnn_results.txt'
    cnn_results = load_cnn_results(cnn_dict_file)
    print('length:', len(cnn_results))