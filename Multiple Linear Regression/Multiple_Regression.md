

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
dataset = pd.read_csv('50-Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values
```


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
```


```python
X = X[: , 1:]

```


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```


```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
y_pred = regressor.predict(X_test)
```


```python
print(y_pred)
```

    [103681.99389669 133055.59979263 133863.15731217  73098.93132588
     179953.41924267 114882.71426648  66826.80454426  97951.73617032
     114958.03502693 169389.34354305]

