
import argparse
from operator import add
import pyspark.pandas as pd
from pyspark.ml.feature import Imputer

from pyspark.sql import SparkSession

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql.functions import *


spark = SparkSession.builder.getOrCreate()


parser = argparse.ArgumentParser()
parser.add_argument("--json_data")
parser.add_argument("--output_data")

args = parser.parse_args()
print(args.json_data)
print(args.output_data)

sc = spark.sparkContext

print(f"Reading JSON from {args.json_data}")

# Loads data.
dataset = spark.read.option("multiline", "true").json(args.json_data)

print(f"Done reading JSON from {args.json_data}")
dataset.printSchema()

print("Getting row count ....")
print(f"Dataset row count: {dataset.count()}")


#dataset.head()
print("Mapping data...")
#features = dataset.rdd.map(lambda x: [float(x["atom-count"])]).toDF()

# def extract_features(atom_count):
#     return float(atom_count)

# extract_features_udf = udf(extract_features)

from pyspark.ml.feature import VectorAssembler 
  
vec_assembler = VectorAssembler(inputCols = ["pubchem-obabel-canonical-smiles"], 
                                outputCol='features') 
  
dataset = vec_assembler.transform(dataset) 

#dataset = dataset.withColumn("features", array(dataset["atom-count"].cast("float")))
print("Printing head features:")
dataset.select("features").show(2)
print("------------")
#dataset = dataset.union(features)
print("Union done!")

dataset.printSchema()

print("Setting up KMeans cluster")
# Trains a k-means model.

kmeans = KMeans().setK(20).setSeed(1)

# kmeans = KMeans().setFeaturesCol(["atom-count", "atomic-numbers", 
#                                 "basis-count", "bond-order", "charge", "cid", 
#                                 "connection-indices", 
#                                 "coordinates", "core-electrons", "dipole-moment", "energy-alpha-gap", 
#                                 "energy-alpha-homo", "energy-alpha-lumo", "energy-beta-gap", 
#                                 "energy-beta-homo", "energy-beta-lumo", "formula", "heavy-atom-count", "homos", "lowdin-partial-charges", 
#                                 "mo-count", "molecular-mass", 
#                                 "mulliken-partial-charges", 
#                                 "multiplicity", 
#                                 #"name", 
#                                 "number-of-atoms", "obabel-inchi", "orbital-energies", "pm6-obabel-canonical-smiles", 
#                                 "pubchem-charge", "pubchem-inchi", "pubchem-isomeric-smiles", "pubchem-molecular-formula", "pubchem-molecular-weight", 
#                                 "pubchem-multiplicity", "pubchem-obabel-canonical-smiles", "pubchem-version", "state", "total-energy"]).setK(2).setSeed(1)
model = kmeans.fit(dataset)

# cluster_id, 

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)