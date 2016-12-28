name := "sfmovies"

version := "1.0"

scalaVersion := "2.11.7"

javacOptions ++= Seq("-source", "1.8", "-target", "1.8")

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "2.0.2",
	"org.apache.spark" %% "spark-sql" % "2.0.2",
	"org.apache.spark" %% "spark-streaming" % "2.0.2",
	"org.apache.spark" %% "spark-mllib" % "2.0.2"
)