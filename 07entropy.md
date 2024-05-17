# 第7章 信息与熵

## 7.1 度量信息


```python
from math import log2
I_red = -log2(4/9)
I_green = -log2(3/9)
I_yellow = -log2(2/9)
print(f"P(red ball)=4/9, information: {round(I_red, 4)} bits")
print(f"P(green ball)=3/9, information: {round(I_green, 4)} bits")
print(f"P(yellow ball)=2/9, information: {round(I_yellow, 4)} bits")
```

    P(red ball)=4/9, information: 1.1699 bits
    P(green ball)=3/9, information: 1.585 bits
    P(yellow ball)=2/9, information: 2.1699 bits
    

## 7.2 信息熵

### 7.2.1 定义


```python
from scipy.stats import entropy

# 概率分布
p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
# 计算熵
e = entropy(p, base=2)
print(f"entropy: {e:.3f}")
```

    entropy: 2.585
    

## 7.2.3 相对熵和交叉熵


```python
from scipy.stats import entropy

p = [9/25, 12/25, 4/25]
q = [1/3, 1/3, 1/3]

d_pq = entropy(p, q, base=2)
d_qp = entropy(q, p, base=2)

print(f"D(P||Q)={d_pq:.4f}")
print(f"D(Q||P)={d_qp:.4f}")
```

    D(P||Q)=0.1231
    D(Q||P)=0.1406
    


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np

x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

logr = LogisticRegression(solver='lbfgs')
logr.fit(x.reshape(-1, 1), y)

y_pred = logr.predict_proba(x.reshape(-1, 1))[:, 1].ravel()
loss = log_loss(y, y_pred)

print('x = {}'.format(x))
print('y = {}'.format(y))
print('Q(y) = {}'.format(np.round(y_pred, 2)))
print('Cross Entropy = {:.4f}'.format(loss))
```

    x = [-2.2 -1.4 -0.8  0.2  0.4  0.8  1.2  2.2  2.9  4.6]
    y = [0. 0. 1. 0. 1. 1. 1. 1. 1. 1.]
    Q(y) = [0.19 0.33 0.47 0.7  0.74 0.81 0.86 0.94 0.97 0.99]
    Cross Entropy = 0.3329
    


```python

```
