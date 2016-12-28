package org.gp.sfmovies

import org.apache.log4j.Level
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}


object Main {

	case class Movie(title: String, releaseYear: Int, locations: String, 
			 funFacts: String, productionCompany: String,
			 distributor: String, director: String, writer: String, 
			 actor1: String, actor2: String, actor3: String)


	val schema = StructType(Array(
		StructField("title", DataTypes.StringType),
		StructField("releaseYear", DataTypes.IntegerType),
		StructField("locations", DataTypes.StringType),
		StructField("funFacts", DataTypes.StringType),
		StructField("productionCompany", DataTypes.StringType),
		StructField("distributor", DataTypes.StringType),
		StructField("director", DataTypes.StringType),
		StructField("writer", DataTypes.StringType),
		StructField("actor1", DataTypes.StringType),
		StructField("actor2", DataTypes.StringType),
		StructField("actor3", DataTypes.StringType)))

	def main(args: Array[String]): Unit = {
		val sparkSession = SparkSession
			.builder()
			.master("local")
			.config("spark.sql.shuffle.partitions", 6)
			.appName("SF Movie Data")
			.getOrCreate()
		sparkSession.sparkContext.setLogLevel(Level.ERROR.toString)
		import sparkSession.implicits._
		val movieDF = sparkSession.read.schema(schema).format("org.apache.spark.csv").option("header", true).csv("sf_film_data.csv").as[Movie]

		//		println("============ Print Schema =================")
		//		root
		//		|-- title: string (nullable = true)
		//		|-- releaseYear: integer (nullable = true)
		//		|-- locations: string (nullable = true)
		//		|-- funFacts: string (nullable = true)
		//		|-- productionCompany: string (nullable = true)
		//		|-- distributor: string (nullable = true)
		//		|-- director: string (nullable = true)
		//		|-- writer: string (nullable = true)
		//		|-- actor1: string (nullable = true)
		//		|-- actor2: string (nullable = true)
		//		|-- actor3: string (nullable = true)

		movieDF.createOrReplaceTempView("moviesVIEW")
		sparkSession.catalog.cacheTable("moviesVIEW")
		val ds = sparkSession.table("moviesVIEW")
		val dsNonNull = ds.na.fill("empty", Seq("locations", "director"))
//		dsNonNull.groupBy($"locations").count().sort($"count".desc).show(false)
		//
		//
		//		val labelIndexer = new StringIndexer().setInputCol("locations").setOutputCol("label")
		//		val ds3 = labelIndexer.fit(ds2).transform(ds2)
		//		ds3.show(false)
		//		println("============ Movies released after 2013 =================")
		//		ds.filter(movieDF.col("releaseYear") > 2013)
		//			.filter(movieDF.col("locations").contains("Bay"))
		//			.distinct()
		//			.show(500, false)
		//

		val indexer = new StringIndexer().setInputCol("locations").setOutputCol("label")
		val li = indexer.fit(dsNonNull).transform(dsNonNull)
		val directorIndexer = new StringIndexer().setInputCol("director").setOutputCol("directorIndexer")
		val di = directorIndexer.fit(li).transform(li)
		val assembler = new VectorAssembler().setInputCols(Array("directorIndexer")).setOutputCol("features")
		val ds2 = assembler.transform(di)
		val lr = new GeneralizedLinearRegression()
			.setFamily("gaussian")
			.setLink("identity")
			.setMaxIter(10)
			.setRegParam(0.3)

		val splits = ds2.randomSplit(Array(0.6, 0.4), seed = 11l)
		val tests = splits(1)
		val training = splits(0)
		val numIterations = 100
		val model = lr.fit(training)
		println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")
		model.transform(tests).sort($"prediction".asc).show()

	}
}
