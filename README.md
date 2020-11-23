# 2020 대경권 대학생 인공지능 프로그래밍 경진대회 ML

[https://programmers.co.kr/competitions/581/dg-univ-2020](https://programmers.co.kr/competitions/581/dg-univ-2020)

# Contest Review

(대회 후기는 블로그에 작성중)

# Data Summary

```python
>>> print(train.shape)
(11769, 15)
>>> print(test.shape)
(5789, 13)
```

```python
>>> train.info()
```

```plain
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 11769 entries, 0 to 11768
Data columns (total 15 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   no         11769 non-null  int64
 1   모델명        11769 non-null  object
 2   연월         11769 non-null  object
 3   연식         11769 non-null  float64
 4   연료         11769 non-null  object
 5   주행거리       11769 non-null  object
 6   인승         765 non-null    float64
 7   최대출력(마력)   10644 non-null  float64
 8   기통         8698 non-null   float64
 9   최대토크(kgm)  10545 non-null  float64
 10  구동방식       10808 non-null  object
 11  자동수동       1313 non-null   object
 12  국산/수입      11769 non-null  object
 13  신차가(만원)    9544 non-null   float64
 14  가격(만원)     11769 non-null  float64
dtypes: float64(7), int64(1), object(7)
memory usage: 1.3+ MB
```

```python
>>> test.info()
```

```plain
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5789 entries, 0 to 5788
Data columns (total 13 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   no         5789 non-null   int64
 1   모델명        5789 non-null   object
 2   연월         5789 non-null   object
 3   연식         5789 non-null   float64
 4   연료         5789 non-null   object
 5   주행거리       5789 non-null   object
 6   인승         389 non-null    float64
 7   최대출력(마력)   5237 non-null   float64
 8   기통         4275 non-null   float64
 9   최대토크(kgm)  5183 non-null   float64
 10  구동방식       5312 non-null   object
 11  자동수동       665 non-null    object
 12  국산/수입      5789 non-null   object
dtypes: float64(5), int64(1), object(7)
memory usage: 588.1+ KB
```

## Translated

```python
>>> train.info()
```

```plain
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 11769 entries, 0 to 11768
Data columns (total 15 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   no         11769 non-null  int64
 1   modelname  11769 non-null  object
 2   ym         11769 non-null  object
 3   yeartype   11769 non-null  float64
 4   fuel       11769 non-null  object
 5   dist       11769 non-null  object
 6   people     765 non-null    float64
 7   power      10644 non-null  float64
 8   cylinder   8698 non-null   float64
 9   torque     10545 non-null  float64
 10  type       10808 non-null  object
 11  mission    1313 non-null   object
 12  country    11769 non-null  object
 13  newprice   9544 non-null   float64
 14  price      11769 non-null  float64
dtypes: float64(7), int64(1), object(7)
memory usage: 1.3+ MB
```

```python
>>> test.info()
```

```plain
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5789 entries, 0 to 5788
Data columns (total 13 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   no         5789 non-null   int64
 1   modelname  5789 non-null   object
 2   ym         5789 non-null   object
 3   yeartype   5789 non-null   float64
 4   fuel       5789 non-null   object
 5   dist       5789 non-null   object
 6   people     389 non-null    float64
 7   power      5237 non-null   float64
 8   cylinder   4275 non-null   float64
 9   torque     5183 non-null   float64
 10  type       5312 non-null   object
 11  mission    665 non-null    object
 12  country    5789 non-null   object
dtypes: float64(5), int64(1), object(7)
memory usage: 588.1+ KB
```

# Kernels

## v1_0

### Preprocessing

- **1 modelname**
  - 맨 앞 한 단어가 브랜드명이라고 가정.
  - train, test 각각 브랜드명 추출 후 합집합 처리하여 columns 수 맞춰주고 범주형으로 one-hot encoding.
- **2 ym**
  - 괄호가 있는 데이터는 괄호 이후 데이터 제거.
  - `'/'` 기준으로 split하여 앞을 `year`, 뒤를 `month`로 두 개의 column 만듦.
  - `year`의 경우 99에서 00 넘어가는 부분을 처리해주어야 하므로, 대충 `30`보다 크면 `19` 붙여서 `19xx` 만들고, 아니면 `20` 붙여서 `20xx` 만들어서 연속형 데이터로 만듦.
- **3 yeartype**
  - 변경 X
- **4 fuel**
  - 범주형 one-hot encoding.
- **5 dist**
  - 중간중간에 `'등록'`이라고 되어 있는 데이터가 있음 (무슨 뜻인지 알아보려 했으나 잘 모르겠음)
  - 형식이 `%d{만|천}{km|ml}` 이므로 전처리 함수 작성하여 수치형으로 변환
  - 결측치는 평균으로 채움
- **6 people**
  - 결측치가 너무 많아 드랍
- **7 power**
  - 결측치만 평균으로 채워서 사용
- **8 cylinder**
  - 결측치는 최빈값으로 채워서 범주형 one-hot encoding.
- **9 torque**
  - 결측치만 평균으로 채워서 사용
- **10 type**
  - 결측치는 최빈값으로 채워서 범주형 one-hot encoding.
- **11 mission**
  - 결측치가 너무 많아 드랍
- **12 country**
  - 쓸 수는 있겠으나, train, test 모두 `'국산'` 데이터밖에 없었다.
  - 분류에 의미가 없으니 드랍
- **13 newprice**
  - 'test에도 없는 데이터를 왜 준 거지? 일단 드랍'

### Algorithms

XGB 0.5 + GB 0.5

```python
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
```

## v1_1

### Preprocessing

- **1 modelname**
  - 드랍

### Algorithms

XGB 0.25 + GB 0.25 + SVR 0.5

```python
from sklearn.svm import SVR

model_svr = SVR(C=1, cache_size=200, coef0=0, degree=3, epsilon=0.0, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

## v1_2

> 피쳐들의 의미도 생각해보자. 전처리를 빡시게 하자.

### Preprocessing

- **2 ym**
  - 명세를 다시 보니 어차피 `yeartype`에 있는 정보와 동일한 정보였다.
  - 드랍
- **4 fuel**
  - train에만 `수소` 레이블이 딱 하나 있는데, 얘를 `기타`로 집어넣었다.
- **6 people**
  - 최대한 살려보려 했으나 결국 드랍
  - 모델명에서 몇인승인지 정보를 빼올 수 있을까 생각했는데 너무 힘들 것 같았다
- **10 type**
  - train에만 3개 존재하는 `RR` 레이블 데이터들은 drop해 버렸다.
- **11 mission**
  - 결측치가 많긴 했는데, 다시 들여다보니 `수동`만 기재되어 있었다.
  - 그럼 기재되지 않은 차들은 다 `자동`이라는 것 아닐까?
  - 결측치들을 모두 `자동`으로 채우고 범주형 one-hot encoding.

### Algorithms

RFR

```python
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=50, random_state=42)
```

## v1_2_1

### Preprocessing

- v1_2와 동일

### Algorithms

XGB 0.5 + LGB 0.5 w/ GridSearchCV

```python
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
import xgboost as xgb
import lightgbm as lgb

def print_best_params(model, params):
    grid_model = GridSearchCV(
        model,
        param_grid = params,
        cv=5,
        scoring='neg_mean_squared_error')

    grid_model.fit(X_train, Y_train)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print(
        '{0} 5 CV 시 최적 평균 RMSE 값 {1}, 최적 alpha:{2}'.format(model.__class__.__name__, np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_
```

```python
xgb_params ={
    'learning_rate': [0.05],
    'max_depth': [5],
    'subsample': [0.9],
    'colsample_bytree': [0.5],
    'silent': [True],
    'gpu_id':[0] ,
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
    'n_estimators':[1000],
    'refit' : [True]
}

xgb_model = xgb.XGBRegressor()
xgb_estimator = print_best_params(xgb_model, xgb_params)

lgb_params = {
    'objective':['regression'],
    'num_leave' : [1],
    'learning_rate' : [0.05],
    'n_estimators':[1000],
    'max_bin' : [80],
    'gpu_id':[0] ,
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
    'refit':[True]
}

lgb_model = lgb.LGBMRegressor()
lgb_estimator = print_best_params(lgb_model, lgb_params)
```

## v1_2_2

### Preprocessing

수치형 데이터 중 `yeartype`, `price`가 skew되어 있는 걸 plot으로 확인했다. `np.log1p()`를 적용하고, 결과값에서 반대로 해 주면 될 것 같았다.

_그런데 지금 보니까 `np.log1p()` 잘 해놓고 예측값에다가 `np.exp()`를 했었다. 원래는 `np.expm1()`이 맞다..._

### Algorithms

XGB 0.5 + LGB 0.5 w/ GridSearchCV

## v1_2_3

> 결측치를 더 잘 채울 것이다  
> MissForest를 쓸 건데  
> 일단 category형 변수로 냅두고 더미 변수를 나중에 만든다

### Preprocessing

결측치들을 단순히 평균/중앙값이 아니라, MissForest 라이브러리를 이용해 채우기로 결정

1. 범주형 Column들의 결측치 처리
   1. `string`형으로 변환
   2. 결측치를 `'NaN'` str로 채움
   3. 다시 `category`형으로 변환
2. 각 범주형 Column마다 LabelEncoder 준비 후 train, test 각각 transform
   1. 이 때 train, test에 각각 다른 LabelEncoder를 쓰지 않도록 주의
   2. transform 후 자료형이 변한다는 말이 있어서 column들을 다시 `category`형으로 변환
3. MissForest 수행
4. Categorical column들의 LabelEncoder inverse transform

(TODO: v1_2_3.ipynb 716번 블록부터 코드)

### Algorithms

XGB 0.5 + LGB 0.5 w/ GridSearchCV
