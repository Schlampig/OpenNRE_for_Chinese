# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
import jieba.posseg as pseg
from flask import Flask, request, jsonify
from prepare import init_embed_and_class, process_data
from config import Config
import models
import ipdb


# Global Initialization
##########################################################################################
app = Flask(__name__)

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
            "score": str(lst_pre[i_sample][1])
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


# Example
##########################################################################################
def run_example():
    s = "《软件体的生命周期》是特德·姜的作品，2015年5月译林出版社出版。译者张博然等。作者特德·姜是为世界科幻界认可的华裔科幻作家。他游走在科幻边缘，在科幻架构上探讨哲学、人性与情感。《软件体的生命周期》一书结集了特德·姜的《软件体的生命周期》、《赏心悦目》、《商人和炼金术士之门》等六部作品。随着数码体市场的发展、壮大、冷淡和萧条，数码体们的命运也随之发生变迁。"
    h = "特德·姜"
    t = "《软件体的生命周期》"
    print("-----example----------------------------------")
    lst_ans = predict(s, h, t)
    for ans in lst_ans:
        for k_ans in ans:
            print(k_ans, " ", ans[k_ans])
        print()
    print("-----example----------------------------------")
    return None


# Demo
##########################################################################################
@app.route("/", methods=["POST"])
def hello():
    context = request.json.get("context")
    head = request.json.get("head")
    tail = request.json.get("tail")
    result = predict(context, head, tail)
    return jsonify({"result": result})


# Run
##########################################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7777, threaded=True)
