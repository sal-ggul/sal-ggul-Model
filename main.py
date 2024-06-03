import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from datetime import datetime, date, timedelta


# yfinance 로 KRW 값 받아와서 반환 해줌
def get_curr_KRW():
    start2 = str(date.today() - timedelta(days=5))
    end2 = str(date.today())

    # 'KRW=X'를 사용하여 환율 데이터를 가져옵니다.
    usd_krw_data = yf.download('KRW=X', start=start2, end=end2)
    return usd_krw_data['Close'][-1]


def show_XY(combined_df):
    # 시각화
    plt.figure(figsize=(16, 8))
    plt.plot(combined_df.index, combined_df['Close'], label='Historical Close Price (ETH-USD)')
    plt.plot(combined_df.index, combined_df['Predicted Close Price'], label='Predicted Close Price (ETH-USD)')
    plt.xlabel('Date')
    plt.ylabel('Close Price KRW')
    plt.title('Ethereum Price Prediction for the Next 3 Months')
    plt.legend()


# get_curr_KRW 로 받아온 KRW를 현재 1 USD 와 곱해줘서 현재 KRW 를 나타내게 함
def cal_Y_KRW():
    # x축과 y축의 눈금 설정
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 1개월 간격 주요 눈금
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # 1개월 간격 보조 눈금
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # 날짜 형식 지정

    curr_KRW = get_curr_KRW()
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int((x * curr_KRW) // 1000000)} M {int((x * curr_KRW) % 1000000 // 1000)} K'))


def predict_Result(model, scaled_data, scaler, df):

    # 데이터의 마지막 365일을 사용하여 향후 3개월을 예측
    last_365_days = scaled_data[-365:]
    x_input = last_365_days.reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    i = 0
    while (i < 90):  # 3개월 예측
        if len(temp_input) >= 360:
            x_input = np.array(temp_input[-360:])
            x_input = x_input.reshape(1, -1, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            break

    predictions = scaler.inverse_transform(lst_output)

    # 예측 시작 날짜 설정
    start_date = df.index[-1] + pd.Timedelta(days=1)

    # 예측된 가격을 날짜와 함께 DataFrame으로 변환
    predicted_df = pd.DataFrame(predictions, columns=['Predicted Close Price'],
                                index=pd.date_range(start=start_date, periods=len(predictions)))

    # 실제 데이터와 예측 데이터 합치기
    combined_df = pd.concat([df[['Close']], predicted_df])

    return combined_df


def init_dataset():
    # 데이터 수집
    df = yf.download('ETH-USD', start='2023-06-04', end='2024-06-04')

    # 데이터 전처리
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 모델 설계
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 모델 컴파일 및 학습
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, batch_size=1, epochs=1)

    return model, scaled_data, scaler, df


# 데이터셋 생성 함수
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


def start_ETH():
    model, scaled_data, scaler, df = init_dataset()
    result = predict_Result(model, scaled_data, scaler, df)
    show_XY(result)
    cal_Y_KRW()

    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()


start_ETH()
