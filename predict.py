# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
import jieba.posseg as pseg
from prepare import init_embed_and_class, process_data
from config import Config
import models
import ipdb


# Global Initialization
##########################################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
best_epoch = 15
num_class = 51
max_length = 120
model_name = "pcnn_att"
word2id_path = "./_data/word2id.pkl"
rel2id_path = "../datasets/DuNRE/relation_nre.json"
embeddings_path = "./_data/vec.npy"

# load embedding matrix
word2id, rel2id = init_embed_and_class(word2id_path, rel2id_path)
embeddings = np.load(embeddings_path)
id2rel = {v: k for k, v in rel2id.items()}

# load model
MODEL = {'pcnn_att': models.PCNN_ATT, 'pcnn_one': models.PCNN_ONE, 'pcnn_ave': models.PCNN_AVE,
      'cnn_att': models.CNN_ATT, 'cnn_one': models.CNN_ONE, 'cnn_ave': models.CNN_AVE}
dict_init = {"word": np.zeros((1, max_length), dtype=np.int64), 
         "pos1": np.zeros((1, max_length), dtype=np.int64), 
         "pos2": np.zeros((1, max_length), dtype=np.int64), 
         "mask": np.zeros((1, max_length, 3), dtype=np.float32),
         "bag_label": list(), "bag_scope": list(), 
         "ins_label": list(), "ins_scope": list()}
model = Config(num_class=num_class, max_length=max_length, is_training=False)
model.load_predict_data(dict_init, embeddings)
model.set_predict_model(model=MODEL[model_name], epoch=best_epoch)



# Sample Processing
##########################################################################################
def get_pair(lst_text, entity, is_head=True):
    # input: lst_text = [(word, word_pos), ...]
    #     entity is a string
    #     is_head is a tag to distinct head or tail (True for head, False for tail)
    # output: lst_pair = [(head, tail), ...]
    assert isinstance(entity, str)
    lst_pair = list()
    if len(entity) > 0:
        for e, ep in lst_text:
            if e == entity:  # entity should not match itself
                continue
            if ('n' in ep) or (ep in ['i', 'j', 's', 'l']):  # select eligible e according to its part-of-speech
                if is_head:
                    lst_pair.append((entity, e))
                else:
                    lst_pair.append((e, entity))
    elif len(entity) == 0:
        lst_entity = [e for e, ep in lst_text if ('n' in ep) or (ep in ['i', 'j', 's', 'l'])]
        lst_entity = list(set(lst_entity))
        for head, tail in itertools.combinations(lst_entity, 2):
            lst_pair.append((head, tail))
            lst_pair.append((tail, head))
    else:
        pass
    lst_pair = list(set(lst_pair))
    return lst_pair


def string2json(text, head, tail):
    # input: text, head, tail, are both string
    # output: sample = [d_1, d_2, ...], where d_i = {sentence: str, head:{word:str, id: str}, relation:empty str}
    assert isinstance(head, str)
    assert isinstance(tail, str)
    assert isinstance(text, str)
    # generate sentence
    lst_text = pseg.lcut(text)
    sentence = [i for i, j in lst_text]
    sentence = " ".join(sentence)
    # generate entity-pairs
    if len(head) > 0 and len(tail) > 0:
        lst_pair = [(head, tail)]
    elif len(head) > 0 and len(tail) == 0:
        lst_pair = get_pair(lst_text, head, True)
    elif len(head) == 0 and len(tail) > 0:
        lst_pair = get_pair(lst_text, tail, False)
    else:
        lst_pair = get_pair(lst_text, "")
    # generate sample with json structure
    sample = list()
    for head, tail in lst_pair:
        d = {
            "sentence": sentence,
            "head": {"word": str(head), "id": str(head)},
            "tail": {"word": str(tail), "id": str(tail)},
            "relation": ""}
        sample.append(d)
    return sample


def prettify(lst_sample, lst_pre):
    # input： lst_sample = [d_1, d_2, ...], where d_i = {sentence: str, head:{word:str, id: str}, relation:empty str}
    #      lst_pre = [sample_1_pre, sample_2_pre, ...], sample_i_pre = (class_name, score)
    # output： lst_answer = [d_1, d_2, ...], where d_i = {"head": str, "tail": str, "relation": str, "score": float}
    lst_answer = list()
    for i_sample, sample in enumerate(lst_sample):
        d = {
            "head": sample["head"]["word"], 
            "tail": sample["tail"]["word"], 
            "relation": lst_pre[i_sample][0], 
            "score": lst_pre[i_sample][1]
            }
        lst_answer.append(d)
    return lst_answer

# Prediction
##########################################################################################
def predict(sentence, head, tail):
    # input: sentence: string
    #     head: string, entity_1
    #     tail: string, entity_2
    # output: lst_answer = [d_1, d_2, ...], where d_i = {"head": str, "tail": str, "relation": str, "score": float}
    global model, rel2id, word2id, embeddings, id2rel
    # initialization
    if not isinstance(sentence, str):
        return []
    if not isinstance(head, str):
        head = ""
    if not isinstance(tail, str):
        tail = ""
    # pre-processing
    lst_sample = string2json(sentence, head, tail)
    dict_res = process_data(lst_sample, rel2id, word2id, is_training=False)
    # extract relation
    model.load_predict_data(dict_res, embeddings)
    lst_pre = model.predict(id2rel)
    # prettify
    lst_answer = prettify(lst_sample, lst_pre)
    return lst_answer


# Run
##########################################################################################
if __name__ == "__main__":
    s = "《后窗》是一部由阿尔弗雷德·希区柯克执导，詹姆斯·斯图尔特、格蕾丝·凯利、瑟尔玛·瑞特等主演的悬疑片。该片讲述了摄影记者杰弗瑞为了消磨时间，于是监视自己的邻居并且偷窥他们每天的生活，并由此识破一起杀妻分尸案的故事。1954年8月1日，该片在美国上映。"
    h = ""
    t = ""
    
    lst_ans = predict(s, h, t)
    for ans in lst_ans:
        for k_ans in ans:
            print(k_ans, " ", ans[k_ans])
        print()
    print("Finished.")

