from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.util import MLUtils

import pandas
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import cross_validation

import numpy
import math


from pyspark.mllib.regression import LabeledPoint

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

pdf = DataFrame.from_csv("train.csv", index_col=None)
df = sqlContext.createDataFrame(pdf)
df = df.map(lambda row: LabeledPoint(row[0], row[1:])).toDF()

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2000).fit(df)
# 0.28756570690198635 with maxCat=4

(trainingData, testData) = df.randomSplit([0.7, 0.3])


# Train a RandomForest model.
rf = RandomForestRegressor(numTrees=20, maxDepth=7, maxBins=1200, featuresCol="indexedFeatures")

# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
# print "Root Mean Squared Error (RMSE) on test data = %g" % rmse

# Compute RMPSE
squares = predictions.rdd.filter(lambda x: x.label != 0).map(lambda x: ((x.label - x.prediction) / x.label) *  ((x.label - x.prediction) / x.label))

mean = squares.mean()
import math
math.sqrt(mean)
