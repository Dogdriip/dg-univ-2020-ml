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
  - ㅇ
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

- XGBoost \* 0.5 + GradientBoost \* 0.5

```python
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
```

```python
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
```
