// Databricks notebook source
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

// COMMAND ----------

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

// COMMAND ----------

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .fit(df)

// COMMAND ----------

val indexed = indexer.transform(df)

// COMMAND ----------


val encoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVec")



// COMMAND ----------

val encoded = encoder.transform(indexed)
encoded.show()