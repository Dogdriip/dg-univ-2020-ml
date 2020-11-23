# 2020 대경권 대학생 인공지능 프로그래밍 경진대회 ML

## Contest Review

(대회 후기는 블로그에 작성중)

## Data Summary

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

### Translated

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

## Kernels

### v1_0
