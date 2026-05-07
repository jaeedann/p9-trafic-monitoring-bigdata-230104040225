from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour
import shutil
import os
import random
from datetime import datetime, timedelta

# =========================
# SETUP PATH (ABSOLUTE)
# =========================
BASE_PATH = os.path.abspath("output")

TRAFFIC_PATH = os.path.join(BASE_PATH, "traffic")
TIME_PATH = os.path.join(BASE_PATH, "traffic_time")
ML_PATH = os.path.join(BASE_PATH, "ml_data")

# =========================
# INIT SPARK
# =========================
spark = SparkSession.builder \
    .appName("Traffic Monitoring UTS") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# =========================
# CLEAN OUTPUT FOLDER
# =========================
if os.path.exists(BASE_PATH):
    shutil.rmtree(BASE_PATH)

os.makedirs(BASE_PATH)

# =========================
# GENERATE DATA (SIMULASI)
# =========================
locations = ["AreaA", "AreaB", "AreaC"]
start_time = datetime.now()

sensor_data = []

for i in range(100):
    for loc in locations:
        sensor_data.append((
            start_time + timedelta(minutes=i),
            loc,
            random.randint(10, 100)
        ))

# =========================
# CREATE DATAFRAME
# =========================
sensor_df = spark.createDataFrame(
    sensor_data,
    ["timestamp", "location", "vehicle_count"]
)

# =========================
# TRANSFORMASI DATA
# =========================

# TOTAL per lokasi
traffic_df = sensor_df.groupBy("location") \
    .sum("vehicle_count") \
    .withColumnRenamed("sum(vehicle_count)", "total_vehicle")

# TREND per waktu
traffic_time_df = sensor_df.groupBy("timestamp") \
    .sum("vehicle_count") \
    .withColumnRenamed("sum(vehicle_count)", "vehicle_count")

# DATA UNTUK ML
ml_df = sensor_df.withColumn("hour", hour(col("timestamp"))) \
    .select("hour", "vehicle_count")

# =========================
# SAVE KE PARQUET
# =========================
traffic_df.write.mode("overwrite").parquet(TRAFFIC_PATH)
traffic_time_df.write.mode("overwrite").parquet(TIME_PATH)
ml_df.write.mode("overwrite").parquet(ML_PATH)

print("✅ SEMUA DATA BERHASIL DISIMPAN")

# =========================
# STOP SPARK
# =========================
spark.stop()
