# 学习Python
*****
## 1、Numpy
__ndarray__：（n维数组对象），一个具有矢量算术运算和复杂广播能力的快速且节省时间的多维数组

ndarray 是一种通用的__同构__数据多维容器 --> 即所有的元素必须数相同类型

```python
data =  [[1,2,3,4],[5,6,7,8,9], dtype=np.int64]
arr = np.array(data)

-> aray([[1,2,3,4],
        [5,6,7,8]])
```

np.array 会尝试为新建的Numpy数组推断出一个合适的数据类型，类型保持在一个特殊的的dtype对象中
```Python
arr.dtype
-> dtype('int64')
```

显式转换
```Python
float_arr = arr.astype(np.float64)
```

广播：不同大小的数组之间的运算


矢量化，不用编写循环就可对数据执行批量运算

__Numpy的设计目标是处理大数据，所以很多情况Numpy的操作很多都是对源数组的视图，即数据不会被复制，视图上的任何修改都会直接反映到源数组上。__

区分如下代码
```Python
arr = np.arange(10)
arr[5:8] = 12  # 切片并且赋值， 会修改原数组

arr2 = arr[5:8].copy() # 显式复制
```

arr2d[0][2] 等价于 arr2d[0,2]


条件逻辑表述为数组运算：
```python
np.where(arr>0, 2, -2)
```
***
## 2、pandas
### Series
Series是一种类似于一维数组的对象，它由一组数据（各种Numpy数据类型）以及一组与之相关的数据标签（即索引）。
```Python
obj = Series([4, 7, -5, 3])

-->
0 4
1 7
2 -5
3 3

索引在左边，值在右边

obj.values  #获取索引
--> array([4, 7, -5, 3])
obj.index  #获取值
--> array([0, 1, 2, 3])

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

obj2['a']  # 通过索引获取Series的单个或者一组值
-->-5
obj2[['c','a']]

obj2[obj2 > 0]
```

Series可以看成是一个定长的有序字典，因为它是索引值到数据值的一个映射
```
'b' in obj2
--> True
```

如果数据被存储在一个Python字典中，可以直接通过这个字典来创建Series
```
dict_data = {'a':1, 'b':2, 'c':3}
obj3 = Series(dict_data)
```

Series对象本身及其索引都有一个name属性
```
obj3.name = 'population'
obj3.index.name = 'state'

obj3.index = ['1', '2', '3']
```


### DataFrame
DataFrame是一个表格型的数据结构

DataFrame既有行索引也有列索引，可以被看做由Series组成的字典（共同用同一个索引）

DataFrame中面向行和面向列的操作基本上是平衡的

DataFrame构建：
```Python
data = {'state' : ['a', 'b', 'c'],
        'year' : ['2010', '2011', '2013'],
        'pop' : [1.3, 2.9, 1.4]}
frame = DataFrame(data)

frame2 = DataFrame(data, columns=['s', 'y', 'p'], index=['one', 'two', 'three'])
-->
       s    y    p
one    a   2010  1.3
two    b   2011  2.9
three  c   2013  1.4


axis=0代表往跨行（down)，而axis=1代表跨列（across)，作为方法动作的副词

换句话说:

>使用0值表示沿着每一列或行标签\索引值向下执行方法

>使用1值表示沿着每一行或者列标签模向执行对应的方法

# 行也可以通过位置或名称的方式进行获取
frame2.ix['three']
-->
s c
y 2013
p 1.4
Name: three
```Python

frame2['s'] 等价于 frame2.s  结果是获取一个Series，而且这个Series拥有原DataFrame相同的索引，且其name属性也已经被相应地设置好了


pop = { 'Nevada': {2001:2.4, 2002:2.9},
        'Ohio': {2000:1.5, 2001:1.7, 2002:3.6} }
frame3 = DataFrame(pop)

# 外层字典的键作为列，里层的键作为行索引
-->
      Nevada  Ohio
2000  NaN     1.5
2001  2.4     1.7
2002  2.9     3.6



frame3.T
```


```python
frame3.index.name = 'year'
frame3.columns.name = 'state'

# values属性会以二维ndarry的形式返回DataFrame中的数据
frame3.values
-->
array([NaN, 1.5],
      [2.4, 1.7],
      [2.9, 3.6])
```

构建Series或DataFrame时，所用到的任何数组或者其他系列的标签都会被转换成一个Index。

Index对象是不可修改的（immutable），即用户不可以修改Index对象，
如 Index[1] = 'b' 错误

Index功能类似一个固定大小的集合

```Python
'Ohio' in frame3.columns
-->True
2003 in frame3.index
-->False

obj.reindex(['a','b','c', 'd'], fill_value=0)  #重新索引，并且缺失值填充

obj.reinde(range(6), method='ffill') #Series的前向填充

frame.index(index=['a', 'b', 'c', 'd'], method='ffill', columns=['Texas', 'Utah', 'California']) #DataFrame的修改行列索引

frame.ix[['a', 'b', 'c', 'd'], ['Texas', 'Utah', 'California']]

```

舍弃指定轴上的项
```Python
new_obj = obj.drop('c')

data.drop(['Colorado', 'Ohio']) # 删index
data.drop(['one', 'two'], axis=1) # 删columns
```

obj.ix[val]  选取DataFrame的单个行或者一组行
obj.ix[:, val] 选取DataFrame的单个列或者列子集
obj.ix[val1, val2] 同时选取行和列

pandas可以对不同的索引的对象进行算术运算。在将对象相加时，如果存在不同的索引对，则结果的索引就是该索引对的__并集__。

自动的数据对齐操作在不重叠的索引处引入了NA值

```Python
# 没有重叠的位置（缺失位置）先补0，然后在进行相加操作
df1.add(df2, fill_value=0)

#重新索引时，指定一个填充值
df1.reindex(columns=df2.columns, fill_value)
```

NumPy的ufuncs(元素级数组方法)也可用于操作pandas对象
```python
np.abs(frame)
```
DataFramede apply方法可以实现__将函数应用到由各列或者行所形成的一维数组中__
```python
f = lambda x: x.max() - x.min()
frame.apply(f)  # 求每一列的最大减去最小

frame.apply(f, axis=1)  # 求每一行的最大减去最小？？？
```
可以返回由多个值组成的Series
```python
def f(x):
  return Series([x.min(), x.max()], index=['min', 'max'])
fram.apply(f)
```
元素级的Python函数
```Python
format = lambda x: '%.2f'% x
frame.applymap(format) # 对dataFrame中的每一个元素进行格式化字符串

frame['e'].map(format) # series的用于应用元素级函数的map方法
```

排序
```Python
frame.sort_index() # 行index排序
frame.sort_index(axis=1， ascending=False) # 列column排序， 降序

frame.sort_index(by= ['b', 'a']) # 按照某几列的值进行排序
```

排名
```Python
obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
-->
0 6.5
1 1.0
2 6.5
3 4.5
4 3.0
5 2.0
6 4.5

obj.rank(method='first') # 破坏平级关系的method 'average', 'min', 'max', 'first'
-->
0 6
1 1
2 7
3 4
4 3
5 2
6 5

frame.rank(ascending=False, axis=1) #对整行进行排序
```

pandas中轴的索引并__不一定是唯一__的！！！
```python
obj = Series(range(5), index=['a', 'a', 'b', 'b']
-->
a 0
a 1
b 2
b 3

obj.index.is_unique
-->False

obj.ix('b')
-->
b 2
b 3
```

python统计分析
```python
df = DataFrame([1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], column=['one', 'two'])

df.mean(axis=1, skipna=False) #按行进行求和，禁用跳过NA
-->
a NaN
b 1.300
c NaN
d -0.275


#返回间接索引，这里是返回最大值的索引
df.idxmax()
-->
one b
two d

#累加型
df.cumsum()
-->
  one  two
a 1.4  NaN
b 8.5  -4.5
c NaN  NaN
d 9.25 -5.8
```

count          非NA值的数量
describe       针对Series或各DataFrame列计算汇总统计
min,max        计算最小值和最大值
argmin,argmax  计算能够获取到最小值和最大值的索引位置（整数）
idxmin,idxmax  计算能够获取到最小值和最大值的索引值
quantile       计算样本的分位数（0到1）
sum            值的总和
mean           值的平均数
median         值的算术中位数（50%分位数）
mad            根据平均值计算平均绝对离差
var            样本值的方差
std            样本值的标准差
skew           样本的偏离（三阶矩）
kurt           样本的峰值（四阶矩）
cumsum         样本的累计和
cummin、cummax 样本值的累计最大值和累计最小值
cumprod        样本值的累计积
diff           计算一阶差分（对时间序列很有用）
pct_change     计算百分数变化


```python
df.corr()  #方差矩阵
df.cov()   #协方差矩阵

#计算df的列？？ 跟另外一个Series，即df2.column2 或者DataFrame之间的相关系数
df.corrwith(df2.column2)
```


```Python
obj = Series(['c', 'a', 'd', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.uniques()
--> array([c, a, d, b], dtype=object)
uniques.sort()

obj.value_counts()  #计算一个Series中各值出现的概率
-->
c 3
a 3
b 2
d 1

#value_counts还有一个顶级pandas方法，可用于任何数组或者序列
pd.value_counts(obj.values, sort=False)

# isin 用于判断矢量化集合的成员资格，可用于选取Series或者DataFrame列中的子集
mask = obj.isin(['b', 'c'])
mask
-->
0 True
1 False
2 False
3 False
4 False
5 True
6 True
7 True
8 True

obj[mask]
-->
0 c
5 b
6 b
7 c
8 c

#对每一列进行value_counts
result = df.apply(pd.value_counts).fillna(0)

***
## 3、matplotlib
