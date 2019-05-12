# OpenNRE for Chinese

## Source:
This work is mainly modified from:
  * **code**: the original tensorflow implemention [OpenNRE](https://github.com/thunlp/OpenNRE) from [Natural Language Processing Lab at Tsinghua University (THUNLP)](https://github.com/thunlp), a pytorch reproduction [OpenNRE-PyTorch](https://github.com/ShulinCao/OpenNRE-PyTorch) from [Shulin Cao](https://github.com/ShulinCao), and another pytorch approach [ChineseNRE](https://github.com/buppt/ChineseNRE) from [buppt](https://github.com/buppt). <br>
  * **literature**: related papers and introduction about Neural Relation Extraction (NRE) are collected [here](https://github.com/Schlampig/Knowledge_Graph_Wander). <br>
  * **old version**: the old version that mainly modified from [OpenNRE](https://github.com/thunlp/OpenNRE) could be found [here](https://github.com/Schlampig/i_learn_deep/tree/master/OpenNRE_thunlp). <br>
  * **embedding dictionary**: [Tencent AI Lab Embedding Corpus for Chinese Words and Phraseshttps](//ai.tencent.com/ailab/nlp/embedding.html). <br>
  * **embedding dictionary example**: Each line of Tencent_AILab_ChineseEmbedding.txt is like \[str(1 dimensions), vec(200 dimensions)\] below:
  ```
  ['的', '0.209092', '-0.165459', '-0.058054', '0.281176', '0.102982', '0.099868', '0.047287', '0.113531', '0.202805', '0.240482', '0.026028', '0.073504', '0.010873', '0.010201', '-0.056060', '-0.063864', '-0.025928', '-0.158832', '-0.019444', '-0.144610', '-0.124821', '0.000499', '-0.050971', '0.113983', '0.088150', '0.080318', '-0.145976', '0.093325', '0.139695', '-0.082682', '-0.034356', '0.061241', '-0.090153', '0.053166', '-0.171991', '-0.187834', '0.115600', '0.219545', '-0.200234', '-0.106904', '0.033836', '0.005707', '0.484198', '0.147382', '-0.165274', '0.094883', '-0.202281', '-0.638371', '-0.127920', '-0.212338', '-0.250738', '-0.022411', '-0.315008', '0.169237', '-0.002799', '0.019125', '0.017462', '0.028013', '0.195060', '0.036385', '-0.051681', '0.154037', '0.214785', '-0.179985', '-0.020429', '-0.044819', '-0.074923', '0.105441', '-0.081715', '-0.034099', '-0.096518', '-0.004290', '0.095423', '0.234515', '-0.138332', '0.134917', '0.082070', '0.051714', '0.159327', '0.061818', '0.037091', '0.239265', '0.073274', '0.170960', '0.223636', '-0.187691', '-0.206850', '-0.051000', '-0.269477', '-0.116970', '0.213069', '-0.096122', '0.035362', '-0.254648', '0.021978', '0.071687', '0.109870', '-0.104643', '-0.175653', '0.097061', '-0.068692', '0.196374', '0.007704', '0.072367', '-0.275905', '0.217282', '-0.056664', '-0.321484', '-0.004813', '-0.041167', '-0.118400', '-0.159937', '0.065294', '-0.092538', '0.013975', '-0.219047', '-0.058431', '-0.177256', '-0.043169', '-0.151647', '-0.006049', '-0.279595', '-0.005488', '0.096733', '0.147219', '0.197677', '-0.088133', '0.053465', '0.038738', '0.059665', '-0.132819', '0.019606', '0.224926', '-0.176136', '-0.411968', '-0.044071', '-0.120198', '-0.107929', '-0.001640', '0.036719', '-0.243131', '-0.273457', '-0.317418', '-0.079236', '0.054842', '-0.143945', '0.168189', '-0.013057', '-0.145664', '0.135278', '0.029447', '-0.141014', '-0.183899', '-0.080112', '-0.113538', '0.071163', '0.134968', '0.141939', '0.144405', '-0.249114', '0.454654', '-0.077072', '-0.001521', '0.298252', '0.160275', '0.085942', '-0.213363', '0.083022', '-0.000400', '0.134826', '-0.000681', '-0.017328', '-0.026751', '0.111903', '0.010307', '-0.124723', '0.031472', '0.081697', '0.071449', '0.011486', '-0.091571', '-0.039319', '-0.112756', '0.171106', '0.026869', '-0.077058', '-0.052948', '0.252645', '-0.035071', '0.040870', '0.277828', '0.085193', '0.006959', '-0.048913', '0.279133', '0.169515', '0.068156', '-0.278624', '-0.173408', '0.035439']
 ```
 
<br>

## Dataset
* **source**: The dataset used for this work is from [BaiDu2019 Relation Extraction Competition](http://lic2019.ccf.org.cn/kg), denoted as DuIE. Note that, rather than directly brought into OpenNRE_for_Chinese, DuIE should be first transformed to dataset DuNRE that has the correct format for the model. <br>
* **format of DuIE**: a sample in DuIE is like: <br>
```
sample = {"postag": [{"word": str, "pos": str}, {"word": str, "pos": str}, ...], 
          "text": str,
          "spo_list": [{"predicate": str, "object_type": str, "subject_type": str, "object": str, "subject": str}, 
                       {"predicate": str, "object_type": str, "subject_type": str, "object": str, "subject": str}, 
                       ...]}
```
* **format of DuNRE**: DuNRE contains three main datasets as follows (factually the same format as [OpenNRE](https://github.com/thunlp/OpenNRE)):
```
1. Sample dataset:
    [
        {
            'sentence': str (with space between word and punctuation),
            'head': {'word': str},
            'tail': {'word': str},
            'relation': str (the name of a class)
        },
        ...
    ]
2. Embeddings dataset:
    [
        {'word': str, 'vec': list of float},
        ...
    ]
            
3. Labels dataset:
    {
        'NA': 0 (it is necessary to denote NA as index 0),
        class_name_1 (str): 1,
        class_name_1 (str): 2,
        ...
    }
```
* **example**: a sample would be like:
```
    [
        {
            'sentence': '《 软件 体 的 生命周期 》 是 美国作家 特德·姜 的 作品 ， 2015 年 5 月 译林 出版社 出版 。 译者 张博然 等 。 ',
            'head': {'word': '特德·姜'},
            'tail': {'word': '软件体的生命周期'},
            'relation': ‘作者’)
        },
        ...
    ]
```

<br>

## Codes Dependency:
```
prepare 
learn  -> config -> train/test
     | -> models -> networks

predict -> prepare
      | -> config -> train/test
      | -> models -> networks        
```

<br>

## Command Line:
* **generate DuNRE**: see [here]().
* **prepare**: transform DuNRE to numpy/pickle file for the model.
```bash
python learn
```
* **train**: train, validate and save the model.
```bash
python learn
```
* **predict**: run the flask server, and predict new samples via Postman.
```bash
python predict
```

<br>

## Requirements
  * Python>=3.5
  * pytorch>=0.3.1
  * scikit-learn>=0.18
  * numpy
  * jieba
  * tqdm
  * Flask(optional, if runing the server.py)
<br>

## TODO
- [x] Design way to automatically generate training samples such as [this strategy]().
- [ ] Try different networks and find the optimal one.

