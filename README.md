# sal-ggul-Model

![이더리움_로고](https://github.com/sal-ggul/sal-ggul-Model/assets/101933437/b5e3ec42-bc46-4d11-bb51-ab4ca99a0acc)

이더리움 구매한 이후로,,, 떨어지기만 하는게 너무 억울해서,, 만들게 된 이더리움 예측 Model 입니다.
이름에서 보이는것처럼 살껄,,,하며 후회하는 모습에 영감을 받아 정해봤습니다.

모델은 LSTM 으로 과거 데이터를 통해 예측 할수있게 해줬고 ETH 의 값은 Yfinance 를 사용해서 가져오게 했습니다.
모두가 성투하시길,,,


# 사용법

> pip install numpy pandas yfinance scikit-learn tensorflow matplotlib

해당 명령어를 통해 코드실행에 필요한 모듈들을 다운받아야 합니다.
이후 repository 에 올라가 있는 main.py 를 다운받아 실행하면 됩니다.

<img width="638" alt="스크린샷 2024-06-05 오전 9 33 33" src="https://github.com/sal-ggul/sal-ggul-Model/assets/101933437/30f129b5-2960-4e4f-ab17-a498ecb50682">

실행하게 되면 지난 1년간 데이터를 받아와 학습을 시작합니다.

<img width="326" alt="스크린샷 2024-06-05 오전 9 35 32" src="https://github.com/sal-ggul/sal-ggul-Model/assets/101933437/a2114973-66e3-4d9b-b2f4-eb34006d38da">

이후 금일 (2024년 6월 4일) 기준 10일 이후의 예측값들을 표시하게 됩니다.

![output](https://github.com/sal-ggul/sal-ggul-Model/assets/101933437/bbbb4335-7938-4522-9807-fc8f49e9423c)

또한 시각화를 위해 3개월 간의 에측값을 그래프로 나타내어 더 명확하게 확인할수 있습니다.


# 해결해야할것

실행을 반복하였을때 결과값이 크게 달라지는 문제가 발생하고 있습니다.
현재 모델은 LSTM 을 사용하고 있지만 모델을 자체 학습해서 사용할 예정입니다.
예측값들을 표시하는 부분에 있어서 사용자가 선택할수 있는 기능을 추가할 예정입니다.


# 주의점

예측값은 과거 데이터를 통해 학습된 값이기에 정확한 결과와 다를수 있습니다..
투자의 책임은 개인에게 있습니다,,,,,

