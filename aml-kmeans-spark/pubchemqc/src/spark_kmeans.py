
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
import logging

handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument("--scaffold_smiles")
parser.add_argument("--output_clusters")

args = parser.parse_args()
print(args.scaffold_smiles)
print(args.output_clusters)

sc = spark.sparkContext

print(f"Reading scaffold_smiles from {args.scaffold_smiles}")

# Loads data.
dataset = spark.read.json(args.scaffold_smiles)

print(f"Done reading scaffold_smiles from {args.scaffold_smiles}")
dataset.printSchema()

print("Getting row count ....")
print(f"Dataset row count: {dataset.count()}")

def make_wide_vector(sz, ones):
    
    try:
        v = [0.0]*sz
        for idx in ones:
            if idx:
                v[idx] = 1.0
            
        return (Vectors.dense(v), 1.0)
    except TypeError as ex:
        print("\n\nError: ", ones, "EXCEPTION: ", ex)
        return (Vectors.sparse(sz, [], []), 1.0)

wide_arrays = dataset.rdd.map(lambda x: make_wide_vector(x["sz"], x["ones"]))

wide_arrays = spark.createDataFrame(wide_arrays, ["features", "weighCol"])

print("Wide Array: ", wide_arrays)

print(wide_arrays.head(3))

print("Executing KMeans clustering")
kmeans = KMeans().setK(2).setSeed(1)

model = kmeans.fit(wide_arrays)

print("KMeans model created: ", model)

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Make predictions
predictions = model.transform(wide_arrays)

print("Predictions: ", predictions)

predictions.write.format('json').save(args.output_clusters)
