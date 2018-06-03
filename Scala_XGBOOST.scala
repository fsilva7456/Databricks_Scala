// Databricks notebook source
// MAGIC %md #Use Global Solutions Database

// COMMAND ----------

// MAGIC %sql
// MAGIC use global_solutions_fs

// COMMAND ----------

// MAGIC %md #Imports

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.{DataUtils, XGBoost}

// COMMAND ----------

// MAGIC %md #Import Data

// COMMAND ----------

val dataset = sqlContext.table("titanic_train")

// COMMAND ----------

dataset.show()

// COMMAND ----------

// MAGIC %md #Vector Assembler

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val assembler =  new VectorAssembler()
  .setInputCols(Array("Pclass", "Sex", "Age", "Fare", "IsAlone"))
  .setOutputCol("features")

val vected = assembler.transform(dataset).withColumnRenamed("PE", "label").drop("Pclass", "Sex", "Age", "Fare", "IsAlone")

// COMMAND ----------

