
# 🐳 Week 01-RNN

### RNN이란?
<img src="https://user-images.githubusercontent.com/33839093/107915160-499a4280-6fa7-11eb-97e9-706075f5f03c.png">

👉 그림에서 초록색이 RNN의 hidden layer에 해당하며, hidden layer는 1층 뿐만 아니라 여러층으로 쌓을 수 있다.
- RNN은 hidden node가 방향을 가진 엣지로 연결되어 순환구조를 이루는 인공신경망의 한 종류
- 순차 데이터(sequential data)는 데이터의 값 뿐만 아니라, 데이터의 순서도 중요함
- 음성, 문자 등 순차적으로 등장하는 데이터 처리에 적합한 모델(기존의 NN, CNN은 sequential data를 처리할 수 없음)
- 시퀀스 길이에 관계없이 input, output을 받아들일 수 있는 네트워크 구조


### 간단한 RNN
<img src="https://user-images.githubusercontent.com/33839093/107910930-cb39a280-6f9e-11eb-8a12-ffe1bb30cd85.jpg" width="700" height="500">

현재 상태인 h_t는 이전 상태인 h_t-1과 입력값 x_t를 받아 갱신된다.

- x_t: input
- h_t: 현재상태의 hidden state
- y_t: 현재상태의 output
- W(hh), W(xh) : 각각의 입력 h_t-1, x_t에 대한 가중치

<img src="https://user-images.githubusercontent.com/33839093/107911253-806c5a80-6f9f-11eb-8868-65185e177d8d.png" width="500" height="150">

👉 d와 D(h)값을 모두 4로 가정했을 때, RNN의 은닉층 연산


### RNN의 활성화 함수 : tanh
<p align="center"><img src="https://user-images.githubusercontent.com/33839093/107911371-ce815e00-6f9f-11eb-9460-17f970b7e576.png" width="100" height="200"></p>

선형함수 h(x) = c*x를 사용해 3층 네트워크를 쌓으면, y(x) = h(h(h(x))) 이므로 y(x) = c*c*c*x와 동일하다.

a = c^3이면 y(x) = a*x와 동일한 식이다. hidden layer가 없는 네트워크로 표현할 수 있기 때문에, 층을 쌓는 효과를 얻고 싶다면 활성화함수로 비선형함수를 사용해야 한다.

(활성화 함수로 주로 tanh함수가 사용되지만, ReLU를 사용하는 시도도 있음)

### RNN의 구조를 직관적으로 이해하기
<img src="https://user-images.githubusercontent.com/33839093/107912196-6c295d00-6fa1-11eb-8858-cd745da91a82.png">

어떤 글자가 주어졌을 때 바로 다음 글자를 예측하는 character-level-model

'h'를 넣으면 'e'
'e'를 넣으면 'l'
'l'를 넣으면 'l'
'l'를 넣으면 'o' 를 예측하는 모델

학습 데이터의 글자는 h, e, l, o 4개로
one-hot vector로 변환하면 [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]로 표현할 수 있다.

첫 번째 항인 h_t-1는 이전 state이므로 값이 없기 때문에 보통 0으로 초기화 한다.


### RNN 실습
- [RNN_hihello.ipynb](https://github.com/HanNayeoniee/NLP-study/blob/main/Week%2001-RNN/Week%2001-RNN_hihello.ipynb) : 다음에 오는 char 예측하기(char단위 예측)
- [RNN_longseq.ipynb](https://github.com/HanNayeoniee/NLP-study/blob/main/Week%2001-RNN/Week%2001-RNN_longseq.ipynb) : 특정 길이의 문장을 사용해 예측하기
- [RNN_timeseries.ipynb](https://github.com/HanNayeoniee/NLP-study/blob/main/Week%2001-RNN/Week%2001-RNN_timeseries.ipynb) : 긴 문장 예측하기

### RNN의 문제점
- 장기의존적(Long-Term Dependency) 문제점
"하늘에 떠있는 구름"이라는 문장을 RNN이 학습한다면 "하늘에 떠있는"으로 "구름"이라는 단어를 유추할 수 있다.
하지만, "나는 한국에서 자랐고 나는 한국어를 유창하게 한다"하는 문장이 있을 때, "한국어를 유창하게 한다"라는 문장을 유추할때 "나는 한국에서 자랐고"와 "나는 한국어를"문장이 비슷하기 때문에 RNN이 두 정보의 문맥을 연결하기 어렵다.
👉 RNN을 깊이 쌓으면 학습시키는데 어려움이 있기 때문에, RNN 대신 LSTM과 GRU를 많이 사용한다.

- Vanishing gradient descent
이전 state -> 다음 state로 넘어갈 때 gradient를 조금만 작게 해도 gradient가 0에 가깝게 되고, gradient를 조금만 크게 해도 gradient가 너무 커진다. gradient를 조절하기 어려움 😥
👉 gradient를 상수로 넘기지 말고 gradient를 조절하는 gate를 달아서 조절하자 = LSTM
gradient를 조절하는 gate에는 단순한 activation function이 아니라 하나의 network가 들어있음

<br>

##### 참고한 링크

- [Sung Kim의 youtube강의 - NN의 꽃 RNN 이야기](https://www.youtube.com/watch?v=-SHPG_KMUkQ)
- [ratsgo's blog - RNN과 LSTM을 이해해보자!](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)
