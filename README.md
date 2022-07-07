# Kaggle Starter Code

In this notebook, we walk through a basic workflow for participating in a kaggle competition. 

Specifically, we will cover:

* Training a basic model on kaggle training data.
* Handling missing values.
* Generate predictions for kaggle test data.
* Save predictions to a `.csv` file for submission.

The Kaggle competition we will be completing is the [Spaceship Titanic](https://www.kaggle.com/t/97bdbfde6b0e4ca6bb42ac1f165bea62). If you do not have a Kaggle account yet, you will need to create one to participate.

Please begin by reviewing the material on the Kaggle competition before following the instructions below.

## Develop a model

#### Import Packages


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
```

Begin by reading in the training data.


```python
df = pd.read_csv('data/train.csv')
df.head()
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
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna().sum()
```




    PassengerId       0
    HomePlanet      157
    CryoSleep       170
    Cabin           153
    Destination     150
    Age             139
    VIP             146
    RoomService     142
    FoodCourt       140
    ShoppingMall    160
    Spa             142
    VRDeck          128
    Name            155
    Transported       0
    dtype: int64



Handle missing values after train test split.

#### Preprocessing

Target variable is `Transported`. Separate target from features and perform train test split.


```python
model_1_df = df.copy()

# Target
y_1 = model_1_df['Transported']

# Single Feature
X_1 = model_1_df[['Spa']]

X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, random_state=42)
```


```python
# Replace missing values with the median
imputer = SimpleImputer(strategy='median')
# Fit imputer to the indepedent variable
# using only the training data
imputer.fit(X_train)
# Replace missing values in the training and test data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

Fit a basic Logistic Regression model


```python
model_1 = LogisticRegression()
model_1.fit(X_train, y_train)
```




    LogisticRegression()



**Evaluate model performance.**


```python
train_preds = model_1.predict(X_train)
test_preds = model_1.predict(X_test)

train_score = accuracy_score(y_train, train_preds)
test_score = accuracy_score(y_test, test_preds)

print('Train score:', train_score)
print('Test score:', test_score)
```

    Train score: 0.6308029487945805
    Test score: 0.6266427718040621


## Create submission predictions

Kaggle competitions will always provide you with a `test` dataset that contains all of the independent variables in the training data, *but does not contain the target column.* 

The idea is that you want to build a model using the training data so it can predict accurately when we do not know the target value.

**Import testing data**


```python
test_df = pd.read_csv('data/tet.csv')
test_df.head(3)
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
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6189_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>G/1004/P</td>
      <td>TRAPPIST-1e</td>
      <td>3.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Eulah Garnes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6354_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1315/P</td>
      <td>TRAPPIST-1e</td>
      <td>48.0</td>
      <td>False</td>
      <td>410.0</td>
      <td>2108.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Megany Carreralend</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1704_02</td>
      <td>Mars</td>
      <td>False</td>
      <td>D/60/S</td>
      <td>55 Cancri e</td>
      <td>18.0</td>
      <td>False</td>
      <td>86.0</td>
      <td>1164.0</td>
      <td>516.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Allota Fincy</td>
    </tr>
  </tbody>
</table>
</div>



Repeat same preprocessing steps as before


```python
test_X = test_df[['Spa']]
```


```python
# Impute using fitted imputer
test_X = imputer.transform(test_X)
```

**Create final predictions**


```python
final_preds = model_1.predict(test_X)
```

**Save predictions**

*The kaggle competition provides the following instructions for submitting predictions:*

----------

Your submission should be in the form a csv file with two columns. 
1. `PassengerId`
2. `Transported`

The `PassengerId` column should be the `PassengerId` column found in the predictors dataset.

**For example,** if I were to submit a csv of predictions where I predict the mean for every observations, the first three rows of the submission would look like this:

| PassengerId    | Transported  |
|-------|---------|
| 0013_01 | True |
| 0018_01	 | False |
| 0019_01	 | True |


***It is recommended that you save your predictions to csv using `pd.to_csv` and that you import the saved file into a notebook, to make sure the file is structured as intended.***

--------

The easiest way to do this, is to add the predictions to the original dataframe and then isolate the columns we want. 


```python
# Add predictions to the test dataframe
test_df['Transported'] = final_preds
# Isolate the columns we want in our submission
submission_df = test_df[['PassengerId', 'Transported']]
```

Check the shape. The shape of our submission *must* be `(2000, 2)`


```python
submission_df.shape
```




    (2000, 2)



**Now we just need to save the submission to a `.csv` file.**

In this case, you should set `index=False`.


```python
submission_df.to_csv('sample_submission.csv', index=False)
```

## Submit Predictions

Once you have saved you predictions to a csv file, you can submit them [here](https://www.kaggle.com/competitions/spaceship-titanic-bsc-ds-2022/)
