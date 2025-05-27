import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

st.title("📍 배송 데이터 군집 분석 (Folium 지도 시각화)")

# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("Delivery.csv")

df = load_data()
st.subheader("📄 데이터 미리보기")
st.dataframe(df)

# 위치 컬럼 확인
lat_col = "Latitude"
lon_col = "Longitude"

if lat_col not in df.columns or lon_col not in df.columns:
    st.error("필수 위치 컬럼 'Latitude' 또는 'Longitude'가 누락되어 있습니다.")
    st.stop()

# 사용자 선택: 군집에 사용할 속성
st.sidebar.header("📊 군집 분석 설정")
numeric_columns = df.select_dtypes(include='number').columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in [lat_col, lon_col]]

selected_features = st.sidebar.multiselect("군집에 사용할 속성 선택", numeric_columns, default=numeric_columns[:2])
n_clusters = st.sidebar.slider("군집 수 (K)", min_value=2, max_value=10, value=3)

if len(selected_features) < 1:
    st.warning("1개 이상의 속성을 선택하세요.")
    st.stop()

# 군집 분석용 데이터 만들기
X = df[selected_features + [lat_col, lon_col]].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[selected_features])

# KMeans 수행
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)
X["Cluster"] = labels

# 중심 좌표 계산
center_lat = X[lat_col].mean()
center_lon = X[lon_col].mean()

# Folium 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
colors = [
    "red", "blue", "green", "purple", "orange", "darkred", 
    "lightblue", "pink", "gray", "cadetblue"
]

for _, row in X.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=5,
        color=colors[int(row["Cluster"]) % len(colors)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Cluster {row['Cluster']}"
    ).add_to(m)

st.subheader("🌍 군집 결과 지도")
st_folium(m, width=700, height=500)
