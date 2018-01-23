###### 房屋数据练习 - 加载数据 ######

# 从HDFS文件中创建RDD
lines = sc.textFile('/home/xxx/i9000-training/spark/house-data/cal_housing.data')
header = sc.textFile('/home/xxx/i9000-training/spark/house-data/cal_housing.domain')
# 查看列信息
header.collect()
# 查看前两条数据
lines.take(2)

###### 房屋数据练习 - 创建DataFrame ######

# 导入相关模块
from pyspark.sql import SQLContext
from pyspark.sql.types import *

# 将原始文本行转为浮点数列表
f = lambda line: [float(sp) for sp in line.split(',')]
rows = lines.map(f)

# 定义schema（DataFrame的结构描述）
names = 'longitude,latitude,housingMedianAge,totalRooms,totalBedrooms,population,households,medianIncome,medianHouseValue'
fields = [StructField(name, FloatType()) for name in names.split(',')]
schema = StructType(fields)

# 创建DataFrame
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(rows, schema)

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

# 房屋数据练习5
from pyspark.mllib.regression import *
to_point = lambda row: LabeledPoint(row[-1], row[0:-1])
points = rows.map(to_point)
model = LinearRegressionWithSGD.train(points)
model.predict(points.first().features)
