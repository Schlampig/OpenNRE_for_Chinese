# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np
from tqdm import tqdm


# Global Initialization
##########################################################################################
def init_embed_and_class(word2id_path, rel2id_path):
    # function to load word2id and rel2id
    # input: word2id_path, rel2id_path are both string
    # output: word2id and rel2id are the same as them in function process_data
    if os.path.isfile(word2id_path):
        with open(word2id_path, "rb") as f:
            word2id = pickle.load(f)
    else:
        raise Exception("[ERROR] word2id file doesn't exist.")

    if os.path.isfile(rel2id_path):
        with open(rel2id_path, "r") as f:
            rel2id = json.load(f)
    else:
        raise Exception("[ERROR] rel2id file doesn't exist.")

    return word2id, rel2id


# Generate Embedding-Matrix
##########################################################################################
def get_embed(word_vec_file_name, save_dir):
    # input: word_vec_file_name is the path string
    #        ori_word_vec = [word_vec_1, word_vec_2, ...], where word_vec_i = {'word': string, 'vec': list(float)}
    # output: word2id = {word: index}
    if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
        raise Exception("[ERROR] Word vector file doesn't exist.")

    print("Loading word_vec_file...")
    with open(word_vec_file_name, "r") as f:
        ori_word_vec = json.load(f)

    print("Building word vector matrix and mapping...")
    word2id = dict()
    word_vec_mat = list()
    word_size = len(ori_word_vec[0]['vec'])
    print("Got {} words of {} dims".format(len(ori_word_vec), word_size))
    for i in tqdm(ori_word_vec):
        word2id[i['word']] = len(word2id)
        word_vec_mat.append(i['vec'])
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    word_vec_mat.append(np.random.normal(loc=0, scale=0.05, size=word_size))  # for UNK
    word_vec_mat.append(np.zeros(word_size, dtype=np.float32))  # for BLANK
    word_vec_mat = np.array(word_vec_mat, dtype=np.float32)

    print("Saving embedding file...")
    np.save(os.path.join(save_dir, "vec.npy"), word_vec_mat)
    with open(os.path.join(save_dir, "word2id.pkl"), "wb") as f:
        pickle.dump(word2id, f)
    return word2id


# Sample Processing
##########################################################################################
def find_pos(sentence, head, tail):
    def find(sentence, entity):
        p = sentence.find(' ' + entity + ' ')
        if p == -1:
            if sentence[:len(entity) + 1] == entity + ' ':
                p = 0
            elif sentence[-len(entity) - 1:] == ' ' + entity:
                p = len(sentence) - len(entity)
            else:
                p = 0
        else:
            p += 1
        return p

    sentence = ' '.join(sentence.split())
    p1 = find(sentence, head)
    p2 = find(sentence, tail)
    words = sentence.split()
    cur_pos = 0
    pos1 = -1
    pos2 = -1
    for i, word in enumerate(words):
        if cur_pos == p1:
            pos1 = i
        if cur_pos == p2:
            pos2 = i
        cur_pos += len(word) + 1
    return pos1, pos2


def process_data(ori_data, rel2id, word2id, max_length=120, is_training=True):
    """
    :param ori_data:
            [
            {
                'sentence': 'Bill Gates is the founder of Microsoft .',
                'head': {'word': 'Bill Gates', 'id': 'm.03_3d', ...(other information)},
                'tail': {'word': 'Microsoft', 'id': 'm.07dfk', ...(other information)},
                'relation': 'founder'
            }, ...
            ]
    :param rel2id:
            {
            'NA': 0,
            'relation_1': 1,
            'relation_2': 2,
            ...}
    :param word2id:
            [
            {'word': 'the', 'vec': [0.418, 0.24968, ...]},
            {'word': ',', 'vec': [0.013441, 0.23682, ...]},
            ...]
    :param max_length: integer, the length of sentence brought into model
    :param is_training: True for training, False for test
    :return: dict_res = {"word": sen_word, "pos1": sen_pos1, "pos2": sen_pos2, "mask": sen_mask,
                         "bag_label": bag_label, "bag_scope": bag_scope, "ins_label": ins_label,
                         "ins_scope": ins_scope}
    """
    # initialization
    sen_tot = len(ori_data)  # number of samples
    sen_word = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_pos1 = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_pos2 = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_mask = np.zeros((sen_tot, max_length, 3), dtype=np.float32)
    sen_label = np.zeros(sen_tot, dtype=np.int64)
    sen_len = np.zeros(sen_tot, dtype=np.int64)
    bag_label = list()
    bag_scope = list()
    bag_key = list()

    print("Processing raw data...")
    for i in tqdm(range(sen_tot)):
        sen = ori_data[i]
        # get class
        if sen['relation'] in rel2id:
            sen_label[i] = rel2id[sen['relation']]
        else:
            sen_label[i] = rel2id['NA']
        # get word indexes
        words = sen['sentence'].split()
        sen_len[i] = min(len(words), max_length)  # length of each sentence
        for j, word in enumerate(words):  # sen_word
            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]  # 第i个样本句子里的第j个词的index
                else:
                    sen_word[i][j] = word2id['UNK']
        for j in range(j + 1, max_length):  # 小于max_length的句子后面补BLANK
            sen_word[i][j] = word2id['BLANK']
        # get position weights
        pos1, pos2 = find_pos(sen['sentence'], sen['head']['word'], sen['tail']['word'])
        if pos1 == -1 or pos2 == -1:
            raise Exception("[ERROR] Position error, index = {}, sentence = {}, "
                            "head = {}, tail = {}".format(i, sen['sentence'], sen['head']['word'], sen['tail']['word']))
        if pos1 >= max_length:
            pos1 = max_length - 1
        if pos2 >= max_length:
            pos2 = max_length - 1
        pos_min = min(pos1, pos2)
        pos_max = max(pos1, pos2)
        for j in range(max_length):
            sen_pos1[i][j] = j - pos1 + max_length  # sen_pos1, sen_pos2
            sen_pos2[i][j] = j - pos2 + max_length
            if j >= sen_len[i]:  # sen_mask
                sen_mask[i][j] = [0, 0, 0]
            elif j - pos_min <= 0:
                sen_mask[i][j] = [100, 0, 0]
            elif j - pos_max <= 0:
                sen_mask[i][j] = [0, 100, 0]
            else:
                sen_mask[i][j] = [0, 0, 100]
        # bag_scope
        if is_training:
            tup = (sen['head']['id'], sen['tail']['id'], sen['relation'])
        else:
            tup = (sen['head']['id'], sen['tail']['id'])
        if bag_key == [] or bag_key[len(bag_key) - 1] != tup:
            bag_key.append(tup)
            bag_scope.append([i, i])
        bag_scope[len(bag_scope) - 1][1] = i
    # bag_label
    print("Processing bag label...")
    if is_training:
        for i in bag_scope:
            bag_label.append(sen_label[i[0]])
    else:
        for i in bag_scope:
            multi_hot = np.zeros(len(rel2id), dtype=np.int64)
            for j in range(i[0], i[1] + 1):
                multi_hot[sen_label[j]] = 1
            bag_label.append(multi_hot)
    # ins_scope
    print("Processing instance label...")
    ins_scope = np.stack([list(range(sen_tot)), list(range(sen_tot))], axis=1)
    # ins_label
    if is_training:
        ins_label = sen_label
    else:
        ins_label = []
        for i in sen_label:
            one_hot = np.zeros(len(rel2id), dtype=np.int64)
            one_hot[i] = 1
            ins_label.append(one_hot)
        ins_label = np.array(ins_label, dtype=np.int64)
    bag_scope = np.array(bag_scope, dtype=np.int64)
    bag_label = np.array(bag_label, dtype=np.int64)
    ins_scope = np.array(ins_scope, dtype=np.int64)
    ins_label = np.array(ins_label, dtype=np.int64)

    dict_res = {"word": sen_word, "pos1": sen_pos1, "pos2": sen_pos2, "mask": sen_mask,
                "bag_label": bag_label, "bag_scope": bag_scope,
                "ins_label": ins_label, "ins_scope": ins_scope}
    return dict_res


# Train and Test(Dev)
##########################################################################################
def get_data(file_name, rel2id_file_name, save_dir, word2id=None, max_length=120, strategy="train"):
    # input: file_name and rel2id_file_name are the path strings for data and class respectively
    #        word2id = {word: index}
    # output: save files
    if strategy not in ["train", "test"]:
        raise Exception("[ERROR] Wrong strategy. (train/test)")
    if file_name is None or not os.path.isfile(file_name):
        raise Exception("[ERROR] Data file doesn't exist.")
    if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
        raise Exception("[ERROR] rel2id file doesn't exist.")
    if len(word2id) == 0 or word2id is None:
        raise Exception("[ERROR] word2id is empty.")

    print("Loading data...")
    with open(file_name, "r") as f:
        ori_data = json.load(f)
    with open(rel2id_file_name, "r") as f:
        rel2id = json.load(f)

    # sorting
    print("Sorting data...")
    ori_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])

    # process data
    tag = True if strategy is "train" else False
    dict_res = process_data(ori_data, rel2id, word2id, max_length=max_length, is_training=tag)

    # saving
    print("Saving files...")
    name_prefix = strategy
    np.save(os.path.join(save_dir, name_prefix + '_word.npy'), dict_res.get("word"))
    np.save(os.path.join(save_dir, name_prefix + '_pos1.npy'), dict_res.get("pos1"))
    np.save(os.path.join(save_dir, name_prefix + '_pos2.npy'), dict_res.get("pos2"))
    np.save(os.path.join(save_dir, name_prefix + '_mask.npy'), dict_res.get("mask"))
    np.save(os.path.join(save_dir, name_prefix + '_bag_label.npy'), dict_res.get("bag_label"))
    np.save(os.path.join(save_dir, name_prefix + '_bag_scope.npy'), dict_res.get("bag_scope"))
    np.save(os.path.join(save_dir, name_prefix + '_ins_label.npy'), dict_res.get("ins_label"))
    np.save(os.path.join(save_dir, name_prefix + '_ins_scope.npy'), dict_res.get("ins_scope"))
    print("Finish saving")
    return None


def init(train_file_name, test_file_name, rel2id_file_name, word_vec_file_name, save_dir, max_length=120):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    word2id = None
    if os.path.isfile(word_vec_file_name):
        word2id = get_embed(word_vec_file_name, save_dir)
    if os.path.isfile(train_file_name):
        get_data(train_file_name, rel2id_file_name, save_dir, word2id=word2id, max_length=max_length, strategy="train")
    if os.path.isfile(test_file_name):
        get_data(test_file_name, rel2id_file_name, save_dir, word2id=word2id, max_length=max_length, strategy="test")
    return None


# Run
##########################################################################################
if __name__ == "__main__":
    init(train_file_name="../datasets/DuNRE/train_nre.json",
        test_file_name="../datasets/DuNRE/dev_nre.json",
        rel2id_file_name="../datasets/DuNRE/relation_nre.json",
        word_vec_file_name="../datasets/DuNRE/word_dictionary_nre.json",
        save_dir="_data",
        max_length=120)
