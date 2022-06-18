### CRASpell
中文纠错模型CRASpell torch版本，
论文为CRASpell: A Contextual Typo Robust Approach to Improve Chinese Spelling Correction


#### 环境依赖
```
torch==1.8.1+cpu
transformers==4.10.1
numpy==1.19.2
```

#### 代码结构
```
 src
 |__common
 |__config
 |__model
 |__utils
 |__train.py
 |__predictv2.py
```


#### 实验结果
```
使用chinese-roberta-wwm-ext预训练模型得到
test data result
token num: gold_n:694, pred_n:783, right_n:586
token check: p=0.748, r=0.844, f=0.793
token correction-1: p=0.954, r=0.805, f=0.873
token correction-2: p=0.714, r=0.805, f=0.757
precision:0.7484026201754532, recall:0.8443791867735061, f1_score:0.7934992640496943

在sighan15中的数据集test数据集上
绝对acc 结果: 0.7372
```

#### 参考文献
[1]: https://github.com/liushulinle/CRASpell