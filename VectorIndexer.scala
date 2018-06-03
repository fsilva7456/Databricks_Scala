// Databricks notebook source
// MAGIC %md #Imports

// COMMAND ----------

import org.apache.spark.ml.feature.VectorIndexer

// COMMAND ----------

val data = spark.read.format("libsvm").load("/FileStore/tables/sample_libsvm_data.txt")

// COMMAND ----------

// MAGIC %md #Create VectorIndexer Object and Set Max Categories to 10

// COMMAND ----------

val indexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexed")
  .setMaxCategories(10)

// COMMAND ----------

// MAGIC %md #Fit Indexer with Data

// COMMAND ----------

val indexerModel = indexer.fit(data)

// COMMAND ----------

val indexedData = indexerModel.transform(data)
indexedData.show()

// COMMAND ----------

display(indexedData)

// COMMAND ----------

