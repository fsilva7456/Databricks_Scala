// Databricks notebook source
// Imports
import spark.implicits._
import scala.util.Try
import sqlContext.implicits._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructType, StructField, LongType}
import org.apache.spark.sql
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.first
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StringIndexer, VectorAssembler, VectorIndexer, StringIndexerModel, Bucketizer}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types._

// COMMAND ----------

libraryDependencies ++= Seq(
// https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.11
  "org.apache.spark" %% "spark-core" % "2.1.1",
// https://mvnrepository.com/artifact/org.apache.spark/spark-sql_2.11
  "org.apache.spark" %% "spark-sql" % "2.1.1"
) 

// COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/linkage")

// COMMAND ----------

val rawblocks = sc.textFile("dbfs:/FileStore/tables/linkage")

// COMMAND ----------

rawblocks.first

// COMMAND ----------

rawblocks.take(20)

// COMMAND ----------

rawblocks.count

// COMMAND ----------

val head = rawblocks.take(10)

// COMMAND ----------

head.length

// COMMAND ----------

head.foreach(println)

// COMMAND ----------

def isHeader(line: String) = line.contains("id_1")

// COMMAND ----------

def isHeader(line: String): Boolean = {
  line.contains("id_1")
}

// COMMAND ----------

head.filter(isHeader).foreach(println)

// COMMAND ----------

head.filter(!isHeader(_)).length

// COMMAND ----------

val noheader = rawblocks.filter(x=> !isHeader(x))

// COMMAND ----------

noheader.first

// COMMAND ----------

noheader.take(10).foreach(println)

// COMMAND ----------

spark.sparkContext

// COMMAND ----------

val prev = spark.read.csv("dbfs:/FileStore/tables/linkage")

// COMMAND ----------

prev.show()

// COMMAND ----------

val parsed = spark.read.
  option("header", "true").
  option("nullValue", "?").
  option("inferSchema", "true").
  csv("dbfs:/FileStore/tables/linkage")

// COMMAND ----------

parsed.show()

// COMMAND ----------

parsed.printSchema()

// COMMAND ----------

parsed.count()

// COMMAND ----------

parsed.cache()

// COMMAND ----------

parsed.count()

// COMMAND ----------

parsed.rdd.
  map(_.getAs[Boolean]("is_match")).
  countByValue()

// COMMAND ----------

parsed.
  groupBy("is_match").
  count().
  orderBy($"count".desc).
  show()

// COMMAND ----------

parsed.agg(avg($"cmp_sex"), stddev($"cmp_sex"), min($"cmp_sex"), max($"cmp_sex")).show()

// COMMAND ----------

parsed.createOrReplaceTempView("linkage")

// COMMAND ----------

spark.sql("SELECT is_match, COUNT(*) cnt FROM linkage GROUP BY is_match ORDER BY cnt DESC").show()

// COMMAND ----------

val summary = parsed.describe()

// COMMAND ----------

summary.show()

// COMMAND ----------

summary.select("summary", "cmp_fname_c1").show()

// COMMAND ----------

val matches = parsed.where("is_match = true")
val matchSummary = matches.describe()

// COMMAND ----------

val misses = parsed.filter($"is_match" === false)
val missSummary = misses.describe()

// COMMAND ----------

missSummary.show()

// COMMAND ----------

summary.printSchema()

// COMMAND ----------

val schema = summary.schema
val longForm = summary.flatMap(row => {
    val metric = row.getString(0)
    (1 until row.size).map(i => {
        (metric, schema(i).name, row.getString(i).toDouble)
    })
})

// COMMAND ----------

val longDF = longForm.toDF("metric", "field", "value")

// COMMAND ----------

longDF.show()

// COMMAND ----------

val wideDF = longDF.
  groupBy("field").
  pivot("metric", Seq("count", "mean", "stddev", "min", "max")).
  agg(first("value"))

// COMMAND ----------

wideDF.select("field", "count", "mean").show()

// COMMAND ----------

def pivotSummary(desc: DataFrame): Dataframe = {
    val schema = desc.schema
    import desc.sparkSession.implicits._
  
    val lf = desc.flatmap(row => {
      val metrics = row.getString(0)
      (1 until row.size).map(i => {
        (metric, schema(i).name, row.getString(i).toDouble)
      })
    }).toDF("metric", "field", "value")
  
    lf.groupBy("field").
      pivot("metric", Seq("count", "mean", "stddev", "min", "max")).
      agg(first("value"))
}

val matchSummaryT = pivotSummary(matchSummary)

// COMMAND ----------



// COMMAND ----------

