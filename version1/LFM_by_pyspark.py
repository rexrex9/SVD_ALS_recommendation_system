__author__='雷克斯掷骰子'
'''
B站:https://space.bilibili.com/497998686
头条:https://www.toutiao.com/c/user/token/MS4wLjABAAAAAxu8A9lNX1qfkRKEyU9Uecqa2opPcZufDLWHbv7m-hVdMVPOe7r_i-k6nw4RY61i/
'''

'''
http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS
'''

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName('alstry').config('spark.executor.memory', '10g').getOrCreate()
df = spark.read.csv('../ml-latest-small/ratings.csv',header=True,schema='user INT, item INT, rating DOUBLE')
als = ALS(rank=100, maxIter=10, seed=0,regParam=0.02,alpha=0.005,userCol="user", itemCol="item")

model = als.fit(df)

model.save('model/spark_lfm.model')



