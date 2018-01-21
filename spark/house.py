# 房屋数据练习1
lines = sc.textFile('/home/xxx/i9000-training/spark/house-data/cal_housing.data')
header = sc.textFile('/home/xxx/i9000-training/spark/house-data/cal_housing.domain')
header.collect()
lines.take(2)

# 房屋数据练习2
from pyspark.sql import SQLContext
from pyspark.sql.types import *
f = lambda line: [float(sp) for sp in line.split(',')]
rows = lines.map(f)
names = 'longitude,latitude,housingMedianAge,totalRooms,totalBedrooms,population,households,medianIncome,medianHouseValue'
fields = [StructField(name, FloatType()) for name in names.split(',')]
schema = StructType(fields)
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(rows, schema)

# 房屋数据练习3
df.describe().show()
df.show(5)
df.printSchema()
df.select('population', 'totalBedrooms').show(5)
df.filter(df['housingMedianAge'] > 40).show(5)
df.groupBy("housingMedianAge").count().show(5)

# 房屋数据练习4
df.select('medianHouseValue').show(5)
from pyspark.sql.functions import *
df = df.withColumn('medianHouseValue', col('medianHouseValue')/100000)
df.select('medianHouseValue').show(5)

# 房屋数据练习5
from pyspark.mllib.regression import *
to_point = lambda row: LabeledPoint(row[-1], row[0:-1])
points = rows.map(to_point)
model = LinearRegressionWithSGD.train(points)
model.predict(points.first().features)
