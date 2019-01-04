import numpy as np

def hamming_score(y_true, y_pred, toggle_output=False):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0])
        set_pred = set( np.where(y_pred[i])[0])
        if toggle_output:
            print('set_true: {0}'.format([id2label[id] for id in set_true]), 'set_pred: {0}'.format([id2label[id] for id in set_pred]))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)
    
def f1(y_true, y_pred):
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0])
        set_pred = set( np.where(y_pred[i])[0] )
        
        correct_preds += len(set_true & set_pred)
        total_preds += len(set_pred)
        total_correct += len(set_true)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return p, r, f1