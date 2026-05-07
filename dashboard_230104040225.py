import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Traffic Dashboard", layout="wide")

st.title("🚦 Smart Traffic Monitoring & Prediction")

BASE_PATH = os.path.abspath("output")

TRAFFIC_PATH = os.path.join(BASE_PATH, "traffic")
TIME_PATH = os.path.join(BASE_PATH, "traffic_time")
ML_PATH = os.path.join(BASE_PATH, "ml_data")

# =========================
# INIT SPARK
# =========================
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("Dashboard") \
        .getOrCreate()

spark = init_spark()

# =========================
# LOAD DATA
# =========================
if not os.path.exists(BASE_PATH):
    st.error("Data belum dibuat! Jalankan main_uts_NIM.py dulu")
    st.stop()

traffic_df = spark.read.parquet(TRAFFIC_PATH).toPandas()
time_df = spark.read.parquet(TIME_PATH).toPandas()
ml_df = spark.read.parquet(ML_PATH).toPandas()

# =========================
# SIDEBAR
# =========================
location = st.sidebar.selectbox(
    "Pilih Lokasi",
    traffic_df["location"].unique()
)

# =========================
# KPI
# =========================
total_all = traffic_df["total_vehicle"].sum()
total_loc = traffic_df[traffic_df["location"] == location]["total_vehicle"].values[0]

col1, col2 = st.columns(2)

col1.metric("Total Semua Kendaraan", total_all)
col2.metric(f"Total {location}", total_loc)

# =========================
# GRAFIK
# =========================
time_df = time_df.sort_values(by="timestamp")

fig = px.line(time_df, x="timestamp", y="vehicle_count", title="Trend Kendaraan")

st.plotly_chart(fig, use_container_width=True)

# =========================
# MACHINE LEARNING
# =========================
X = ml_df[["hour"]]
y = ml_df["vehicle_count"]

model = LinearRegression()
model.fit(X, y)

# Slider prediksi
hour_input = st.slider("Prediksi Jam", 0, 23, 12)

prediction = model.predict(np.array([[hour_input]]))[0]

st.subheader("🤖 Prediksi Kendaraan")
st.metric("Jumlah Kendaraan (Prediksi)", int(prediction))
