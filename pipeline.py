#Ahmed Negm December 24, 2023

import findspark 
findspark.init() #Findspak simplifies the process of using Apache Spark with python

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import LineaRegression
from pyspark.ml import pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.appName("Airfoil Noise Prediction").getOrCreate() #start the connection

df = spark.read.csv("NASA.csv", header=True, inferSchema=True) #load the dataset 

df.show(5) #print the top 5 rows of the dataset

rowcount1 = df.count() #printing the total number of rows in the dataset

df.drop_duplicates() #drop the duplicate rows
df.dropna() #dropping all the empty rows

df = df.withColumnRenamed("SoundLevel", "SoundLevelDecibels") #renaming the column SoundLevel to SoundLevelDecibles

df.write.parquet("NASA.parquet") #saving the dataframe as NASA airfoil Noise cleaned

df = spark.read.parquet("NASA.parquet") #loading the data from NASA.parquet file into dataframe

df.count() #printing the total number of rows in the dataset

#define the vector assembler pipeline stage
#stage 1 - assembling the input columsn into a single column "features" use all the columns except SoundLevelDecibles as input feature

input_features = [column for column in df.columns if column!= "SoundLevelDecibels"]
assembler = VectorAssembler(inputCols=input_features, outputCol="features")

#stage 2 of the pipline
#scaling the features using StandardScaler 
scaler = StandardScaler(inputCol= "features", outputCol="scaledFeaturs")

#stage 3 creating a linear Regression stage to predict the soundLevels
lr = LineaRegression(featuresCol="scaledFeatures", labelCol="SoundLevelDecibels")

#NOW building the pipleine with the above 3 stages
pipline = pipeline(stages=[assembler, scaler, lr])

#now splitting the data with 70:30 split
(trainingData, testingData) = df.randomSplit([0.7,0.3], seed=42)

#task 8 fitting the pipline
pipeline = pipeline.fit(trainingData)

#part 3 predicting using the model
predictions = pipelineModel.transform(testingData)

#printing the Mse
mseEvalulator = RegressionEvaluator(
    labelCol="SoundLevelDecibels"
    predictoinCol="Prediction",
    metricName="mse"
)
mse = mseEvalulator.evaluate(predictions)
print(mseEvalulator)

#PRINTING THE R2
r2evalulator = RegressionEvaluator(
    labelCol="SoundLevelDecibels"
    predictoinCol="Prediction",
    metricName="r2"
)
r2 = r2evalulator.evaluate(predictions)
print(r2)

#Persisting the model
pipelineModel.write().save("NASA Prediction")

#loading the model from the path "Final Project"
loadedpipelinemodel = piplineModel.load("NASA Prediction")

#make predictions using the loaded model on the test data
predictions = loadedPipelineModel.transform(testingData)

#showing predictions
predictions.select("SoundLevelDecibels", "prediction").show(5)
