# -*- coding: utf-8 -*-
"""
author: Kyle Chen
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 設置頁面布局為寬模式
st.set_page_config(layout="wide")

# 創建列佈局
col1, col2 = st.columns([1, 3])

# 左側選單
with col1:
    st.markdown("<h2 style='display: inline-block; vertical-align: middle;'>💰金融商品價格預測</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px;'>使用 ARIMA 模型來預測未來的金融商品價格📈</p>", unsafe_allow_html=True)

    # 讓用戶選擇金融商品和預測周期
    product = st.selectbox('選擇金融商品', ['黃金 (GLD)', '原油 (USO)'])
    period = st.selectbox('選擇預測周期', ['日資料', '月資料'])
    criterion = st.selectbox('選擇模型評估標準', ['AIC', 'BIC'])
    alpha = st.selectbox('選擇信賴區間的alpha值', [0.2, 0.1, 0.05])

    # 添加確認按鈕
    if st.button('確認選擇'):
        # 設置代碼和數據抓取區間
        if product == '黃金 (GLD)':
            ticker = 'GLD'
            product_name = '黃金'
        else:
            ticker = 'USO'
            product_name = '原油'

        if period == '日資料':
            start_date = datetime.today() - timedelta(days=365)
            forecast_days = 7
            resample_freq = None
            title = '日資料'
        else:
            start_date = datetime.today() - timedelta(days=3100)
            forecast_months = 12
            resample_freq = 'M'
            title = '月資料'

        # 抓取資料
        st.write(f'正在從 Yahoo Finance 上抓取最新的{product_name}價格資料...')
        try:
            data = yf.download(ticker, start=start_date, end=datetime.today() - timedelta(days=1))
            if data.empty:
                raise ValueError("抓取的資料為空。")
            data.dropna(inplace=True)
        except Exception as e:
            st.error(f'無法從 Yahoo Finance 抓取 {ticker} 的資料，請稍後再試。錯誤信息: {e}')
            st.stop()

        # 聚合資料（如果需要）
        if resample_freq:
            try:
                data = data.resample(resample_freq).mean()
                # 無條件取得聚合後的月資料的結束月份為前一個月
                data = data.iloc[:-1]
            except Exception as e:
                st.error(f'聚合資料時發生錯誤: {e}')
                st.stop()

        # 設置差分
        y = data['Adj Close']

        # 初始化最佳參數和最佳標準
        best_criterion = np.inf
        best_order = None
        best_seasonal_order = None

        # ARIMA 參數範圍
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        P_values = range(0, 2)
        D_values = range(0, 2)
        Q_values = range(0, 2)

        def evaluate_arima_model(order, seasonal_order):
            try:
                model = sm.tsa.SARIMAX(y, order=order, seasonal_order=seasonal_order,
                                       enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False)
                if criterion == 'AIC':
                    return result.aic, order, seasonal_order
                else:
                    return result.bic, order, seasonal_order
            except:
                return np.inf, order, seasonal_order

        # 使用 ThreadPoolExecutor 進行多執行緒運行
        try:
            with ThreadPoolExecutor() as executor:
                futures = []
                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            for P in P_values:
                                for D in D_values:
                                    for Q in Q_values:
                                        order = (p, d, q)
                                        seasonal_order = (P, D, Q, 12)
                                        futures.append(executor.submit(evaluate_arima_model, order, seasonal_order))

                for future in futures:
                    crit, order, seasonal_order = future.result()
                    if crit < best_criterion:
                        best_criterion = crit
                        best_order = order
                        best_seasonal_order = seasonal_order
        except Exception as e:
            st.error(f'評估 ARIMA 模型時發生錯誤: {e}')
            st.stop()

        # 使用最佳參數建立 ARIMA 模型
        try:
            model = sm.tsa.SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order,
                                   enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False)
        except Exception as e:
            st.error(f'建立 ARIMA 模型時發生錯誤: {e}')
            st.stop()

        # 預測未來的價格
        try:
            forecast_steps = forecast_days if resample_freq is None else forecast_months
            forecast = result.get_forecast(steps=forecast_steps)

            # 生成非周末日期
            if resample_freq is None:
                forecast_index = []
                next_date = data.index[-1]
                while len(forecast_index) < forecast_steps:
                    next_date += timedelta(days=1)
                    if next_date.weekday() < 5:  # 周一到周五
                        forecast_index.append(next_date)
                forecast_index = pd.to_datetime(forecast_index)
            else:
                forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq=resample_freq)[1:]
        except Exception as e:
            st.error(f'預測未來價格時發生錯誤: {e}')
            st.stop()

        # 預測區間
        try:
            pred_ci = forecast.conf_int(alpha=alpha)
        except Exception as e:
            st.error(f'計算預測區間時發生錯誤: {e}')
            st.stop()

        with col2:
            # 建立圖表
            fig = go.Figure()

            # 歷史數據
            fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='歷史數據'))

            # 預測數據
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast.predicted_mean, mode='lines', name='預測數據'))

            # 預測區間
            fig.add_trace(go.Scatter(x=forecast_index, y=pred_ci.iloc[:, 0], fill=None, mode='lines', line=dict(color='lightgrey'), name='預測區間下限'))
            fig.add_trace(go.Scatter(x=forecast_index, y=pred_ci.iloc[:, 1], fill='tonexty', mode='lines', line=dict(color='lightgrey'), name='預測區間上限'))

            # 設定圖表
            fig.update_layout(title=f'{product_name}價格預測 ({title})', xaxis_title='日期', yaxis_title='價格')

            # 顯示圖表
            st.plotly_chart(fig, use_container_width=True)

            # 創建新的列佈局來水平排列表格
            table_col1, table_col2 = st.columns(2)

            if resample_freq:
                # 顯示聚合後月資料的表格
                with table_col1:
                    st.write('聚合後月資料:')
                    st.write(data.tail(12))  # 最後12個月的資料

                # 顯示預測的後資料的表格
                with table_col2:
                    st.write(f'預測的後{forecast_steps}月資料:')
                    forecast_data = pd.DataFrame({'Date': forecast_index, 'Predicted Adj Close': forecast.predicted_mean})
                    forecast_data.set_index('Date', inplace=True)
                    st.write(forecast_data)
            else:
                # 顯示最近的7天資料的表格
                with table_col1:
                    st.write('最近的7天資料:')
                    st.write(data.tail(7))  # 最後7天的資料

                # 顯示預測的後7天資料的表格
                with table_col2:
                    st.write('預測的後7天資料:')
                    forecast_data = pd.DataFrame({'Date': forecast_index, 'Predicted Adj Close': forecast.predicted_mean})
                    forecast_data.set_index('Date', inplace=True)
                    st.write(forecast_data)