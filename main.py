import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.title("📦 배송 데이터 군집 분석")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("Delivery.csv")
    return df

df = load_data()
st.subheader("데이터 미리보기")
st.dataframe(df)

# 숫자형 변수 선택
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# 사용자로부터 군집에 사용할 열 선택
st.sidebar.header("군집 설정")
selected_features = st.sidebar.multiselect("군집에 사용할 열 선택", numeric_cols, default=numeric_cols[:2])
n_clusters = st.sidebar.slider("군집 수 (K)", min_value=2, max_value=10, value=3)

# 군집 분석 실행
if len(selected_features) >= 2:
    # 데이터 정규화
    X = df[selected_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans 모델 학습
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # 결과를 데이터프레임에 추가
    X_result = X.copy()
    X_result["Cluster"] = labels

    # 시각화 (2D로만 지원)
    if len(selected_features) >= 2:
        fig = px.scatter(
            X_result, 
            x=selected_features[0], 
            y=selected_features[1], 
            color=X_result["Cluster"].astype(str),
            title=f"KMeans 군집화 결과 (K={n_clusters})",
            symbol="Cluster"
        )
        st.plotly_chart(fig)
else:
    st.warning("2개 이상의 숫자형 열을 선택하세요.")
