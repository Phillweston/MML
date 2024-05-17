# 第 1 章 向量和向量空间

## 1.1 向量

### 1.1.1 描述向量

《跟老齐学 Python：数据分析》

《Python 大学实用教程》

《数据准备和特征工程》


```python
import numpy as np
u = np.array([1, 6, 7])
u
```




    array([1, 6, 7])




```python
v = u.reshape(-1, 1)
v
```




    array([[1],
           [6],
           [7]])




```python
import random, time

# 创建一个列表
lst = [random.randint(1, 100) for i in range(100000)]


start = time.time()
lst2 = [i*i for i in lst]    # 用列表解析的方式计算每个数的平方
end = time.time()
print(f"列表解析用时： {end - start}s")

vlst = np.array(lst)    # 将列表转换为数组表示的向量
start2 = time.time()
vlst2 = vlst * vlst    # 用数组相乘计算每个数的平方
end2 = time.time()
print(f"数组（向量）运算用时：{end2 - start2}s")
print(f"列表解析的运算时间是向量运算时间的：{round((end-start)/(end2-start2), 3)}倍")
```

    列表解析用时： 0.008679866790771484s
    数组（向量）运算用时：0.0005750656127929688s
    列表解析的运算时间是向量运算时间的：15.094倍
    

**应用：NLP**

d1: mathematics machine learn

d2: learn python learn mathematics

![](./lessonimages/ml.png)

$\pmb{d}_1 = \begin{bmatrix}1&1&1&0\end{bmatrix}$，$\pmb{d}_2 = \begin{bmatrix}2&0&1&1\end{bmatrix}$

**词袋**


```python
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer=CountVectorizer()                                              # (1)
corpus=["mathematics machine learn", "learn python learn mathematics"]    # (2)
cor_vec = vectorizer.fit_transform(corpus)                                # (3)
vectorizer.get_feature_names()                                            # (4)
```




    ['learn', 'machine', 'mathematics', 'python']




```python
print(cor_vec)
```

      (0, 2)	1
      (0, 1)	1
      (0, 0)	1
      (1, 2)	1
      (1, 0)	2
      (1, 3)	1
    


```python
import pandas as pd
df = pd.DataFrame(cor_vec.toarray(), columns=vectorizer.get_feature_names())
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>learn</th>
      <th>machine</th>
      <th>mathematics</th>
      <th>python</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**字词频数** $n_{ij}$

**字词频率** tf

**逆向文件频率** idf

tf-idf


```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans = TfidfTransformer()
tfidf = tfidf_trans.fit_transform(cor_vec)

# 每个字词的idf
tfidf_trans.idf_
```




    array([1.        , 1.40546511, 1.        , 1.40546511])




```python
tfidf.toarray()
```




    array([[0.50154891, 0.70490949, 0.50154891, 0.        ],
           [0.75726441, 0.        , 0.37863221, 0.53215436]])



### 1.1.2 向量的加法


```python
import numpy as np
np.array([[2],[1]]) + np.array([[3], [3]])
```




    array([[5],
           [4]])




```python
np.array([[2],[1]]) - np.array([[3], [3]])
```




    array([[-1],
           [-2]])



**1.1.3 向量的数量乘法**


```python
2 * np.array([[3], [1]])
```




    array([[6],
           [2]])




```python
-1 * np.array([[3], [1]])
```




    array([[-3],
           [-1]])




```python

```
