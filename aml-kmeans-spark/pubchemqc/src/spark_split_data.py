import argparse

from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()


parser = argparse.ArgumentParser()
parser.add_argument("--input_data")
parser.add_argument("--num_partitions")
parser.add_argument("--partitions")

args = parser.parse_args()
print(args.input_data)
print(args.num_partitions)
print(args.partitions)

num_partitions = int(args.num_partitions)

sc = spark.sparkContext

print(f"Reading CSV from {args.input_data}")

# Loads data.
dataset = spark.read.csv(args.input_data)

print(f"Done reading CSV from {args.input_data}")
dataset.printSchema()

dataset.repartition(num_partitions).write.mode('overwrite').csv(args.partitions)

print("Saved data to", args.partitions)