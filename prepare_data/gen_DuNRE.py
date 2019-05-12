import os
import json
import jieba
from tqdm import tqdm
import ipdb


"""
data DuIE:
{"postag": [{"word": str, "pos": str}, {"word": str, "pos": str}, ...], 
 "text": str,
 "spo_list": [{"predicate": str, "object_type": str, "subject_type": str, "object": str, "subject": str}, 
              {"predicate": str, "object_type": str, "subject_type": str, "object": str, "subject": str}, 
              ...]}
                                      
data DuNRE:
[
    {
        'sentence': 'Bill Gates is the founder of Microsoft .',
        'head': {'word': 'Bill Gates', ...(other information)},
        'tail': {'word': 'Microsoft', ...(other information)},
        'relation': 'founder'
    },
    ...
]

data embeddings:
[
    {'word': 'the', 'vec': [0.418, 0.24968, ...]},
    {'word': ',', 'vec': [0.013441, 0.23682, ...]},
    ...
]
            
data relation classes:
{
    'NA': 0
    'founder': 1
    ...
}

"""


def text2sentence(text):
    lst_text = jieba.lcut(text)
    sentence = " ".join(lst_text)
    return sentence


def DuIE2DuNRE_data(load_path, save_path):    
    lst_new = list()
    with open(load_path) as f:
        for line in tqdm(f):
            sample = json.loads(line)
            text = sample.get("text")
            sentence = text2sentence(text)
            lst_spo = sample.get("spo_list")
            for spo in lst_spo:
                # 原数据集中，机构与组织的谓词都是“成立日期”，需要区分
                r = "创办日期" if spo.get("subject_type")=="机构" else spo.get("predicate")
                h = spo.get("subject")
                t = spo.get("object")
                d = {"sentence": sentence,
                     "head": {"word": h, "id": h},
                     "tail": {"word": t, "id": t},
                     "relation": r}
                lst_new.append(d)
                
    with open(save_path, "w") as f:
        json.dump(lst_new, f)
    print("Succeed to save.")
    return None
            

def schema2relation(load_path, save_path):
    dict_relation = dict()
    with open(load_path) as f:
        for i_line, line in enumerate(tqdm(f)):
            schema = json.loads(line)
            # 原数据集中，机构与组织的谓词都是“成立日期”，需要区分
            if (schema.get("predicate") == "成立日期") and (schema.get("subject_type") == "机构"):
                dict_relation["创办日期"] = int(i_line+1)
            else:
                dict_relation[schema.get("predicate")] = int(i_line+1)                
        dict_relation["NA"] = 0
    
    with open(save_path, "w") as f:
        json.dump(dict_relation, f)
    print("Succeed to save.")
    return None


def EmbedingMat2EmbedingJson(load_path, save_path):
    # Too large to store, abandoned
    lst_embed = list()
    with open(load_path, "r") as f:
        for line in tqdm(f):
            array = line.split()
            word = "".join(array[0: -200])
            vector = list(map(float, array[-200:]))
            d = {"word": word, "vec": vector}
            lst_embed.append(d)
    print("Begin to save...")
    with open(save_path, "w") as f:
        json.dump(lst_embed, f)
    print("Succeed to save.")
    return None
        

if __name__ == "__main__":
    root_path = "./DuNRE"
    print("Transform dev data...")
    DuIE2DuNRE_data("./DuIE/dev_data.json", os.path.join(root_path, "dev_nre.json"))
    print("Transform training data...")
    DuIE2DuNRE_data("./DuIE/train_data.json", os.path.join(root_path, "train_nre.json"))
    print("Transform relation data...")
    schema2relation("./DuIE/all_50_schemas", os.path.join(root_path, "relation_nre.json"))
#     print("Transform embedding data...")
#     EmbedingMat2EmbedingJson("./Tencent_AILab_ChineseEmbedding.txt", os.path.join(root_path, "word_vec_nre.json"))
