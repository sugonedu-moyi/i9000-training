###### 练习 - Spark版单词计数 ######
lines = sc.textFile('hdfs://XXX')
print(lines.take(2))
f = lambda line: [(str(word).lower(), 1) for word in line.split()]
words = lines.flatMap(f)
print(words.take(3))
word_count_rdd = words.reduceByKey(lambda total, count: total + count)
print(word_count_rdd.take(3))
word_to_count = dict(word_count_rdd.collect())
print(word_to_count['spark'])

###### 房屋数据练习 - 加载数据 ######

# 从HDFS文件中创建RDD
lines = sc.textFile('XXX')
# 查看前两条数据
print(lines.take(2))

###### 房屋数据练习 - 创建DataFrame ######

# 导入相关模块
from pyspark.sql import SQLContext

# 将原始文本行转为浮点数列表
f = lambda line: [float(sp) for sp in line.split(',')]
rows = lines.map(f)

# 9个字段的名称
fields = ['longitude','latitude','housingMedianAge','totalRooms','totalBedrooms','population','households','medianIncome','medianHouseValue']

# 创建DataFrame
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(rows, fields)

###### 房屋数据练习 - DataFrame相关操作 ######

# 显示数据摘要
df.describe().show()
# 显示DataFrame的前5条
df.show(5)
# 显示DataFrame的schema
df.printSchema()
# 显示人口数量和卧室数量两列内容
df.select('population', 'totalBedrooms').show(5)
# 选择年龄中位数超过40的区块组
df.filter(df['housingMedianAge'] > 40).show(5)
# 按年龄中位数分组统计区块组的数量
df.groupBy("housingMedianAge").count().show(5)

from pyspark.sql.functions import *
# 查看调整前的目标列
df.select('medianHouseValue').show(5)
# 调整为以10万为单位
df = df.withColumn('medianHouseValue', col('medianHouseValue')/100000)
# 查看调整后的目标列
df.select('medianHouseValue').show(5)

###### 房屋数据练习 - Spark线性回归练习 ######

# 导入相关的模块
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.linalg import Vectors

# 定义一个函数：将一行（一个区块组）转换为适合训练和预测的格式
f = lambda row: [row[-1], Vectors.dense(row[0:-1])]

# 创建用于机器学习的房屋数据DataFrame
ml_df = sqlContext.createDataFrame(rows.map(f), ['label', 'features'])
ml_df.show(3)

# 将数据划分为训练集和测试集
train_data, test_data = ml_df.randomSplit([0.8, 0.2])
# 线性回归算法对象
lr = LinearRegression()
# 在训练集上训练得到模型
model = lr.fit(train_data)
# 使用模型在测试集上做预测
predict_df = model.transform(test_data)
predict_df.show(3)

