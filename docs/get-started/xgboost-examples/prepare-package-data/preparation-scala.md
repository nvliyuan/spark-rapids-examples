## Prepare packages and dataset for scala

For simplicity export the location to these jars. All examples assume the packages and dataset will be placed in the `/opt/xgboost` directory:

### Download the jars

1. Download the RAPIDS Accelerator for Apache Spark plugin jar
   * [RAPIDS Spark Package](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/22.02.0/rapids-4-spark_2.12-22.02.0.jar)
  
   Then download the version of the cudf jar that your version of the accelerator depends on.

     * [cuDF Package](https://repo1.maven.org/maven2/ai/rapids/cudf/22.02.0/cudf-22.02.0-cuda11.jar)

### Build XGBoost Scala Examples

Following this [guide](/docs/get-started/xgboost-examples/building-sample-apps/scala.md), you can get *sample_xgboost_apps-0.2.2-jar-with-dependencies.jar* and copy it to `/opt/xgboost`

### Download dataset

You need to download mortgage dataset to `/opt/xgboost` from this [site](https://docs.rapids.ai/datasets/mortgage-data)
, download Taxi dataset from this [site](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
, download Agaricus dataset from this [site](https://gust.dev/r/xgboost-agaricus).

### Setup environments

``` bash
export SPARK_XGBOOST_DIR=/opt/xgboost
export CUDF_JAR=${SPARK_XGBOOST_DIR}/cudf-22.02.0-cuda11.jar
export RAPIDS_JAR=${SPARK_XGBOOST_DIR}/rapids-4-spark_2.12-22.02.0.jar
export SAMPLE_JAR=${SPARK_XGBOOST_DIR}/sample_xgboost_apps-0.2.2-jar-with-dependencies.jar
```