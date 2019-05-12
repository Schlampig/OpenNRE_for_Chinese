import json
from collections import Counter
from tqdm import tqdm

load_train_path = "./DuIE/train_data.json"
load_dev_path = "./DuIE/dev_data.json"
lst_all = list()
print("Read train data ...")
with open(load_train_path) as f:
    for line in tqdm(f):
        line = json.loads(line)
        lst_pos = line.get("postag")
        lst_pos = [pos.get("word") for pos in lst_pos]
        lst_all.extend(lst_pos)
print("Read dev data ...")
with open(load_dev_path) as f:
    for line in tqdm(f):
        line = json.loads(line)
        lst_pos = line.get("postag")
        lst_pos = [pos.get("word") for pos in lst_pos]
        lst_all.extend(lst_pos)
           
dict_all = dict(Counter(lst_all))
print(len(dict_all))


print("Check vector ...")
load_embed_path = "./Tencent_AILab_ChineseEmbedding.txt"
save_path = "./DuNRE/word_dictionary_nre.json"
lst_embed = list()
with open(load_embed_path, "r") as f:
    for line in tqdm(f):
        array = line.split()
        word = "".join(array[0: -200])
        if word in dict_all:
            vector = list(map(float, array[-200:]))
            d = {"word": word, "vec": vector}
            lst_embed.append(d)
print("Begin to save...")
with open(save_path, "w") as f:
    json.dump(lst_embed, f)
print("Succeed to save.")
