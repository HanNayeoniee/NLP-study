
# 🦜 Week 02-LSTM

### LSTM이 필요한 이유
- RNN은 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 역전파시 gradient가 점차 줄어들어 학습능력이 크게 저하되는 문제(vanishing gradient problem)가 있다.
- 이 문제를 극복하기 위해 고안된 것이 LSTM, LSTM은 RNN의 hidden state에 cell-state를 추가한 구조이다.


### LSTM(Long Short Term Memory)
<img width="424" alt="rnn_vs_lstm" src="https://user-images.githubusercontent.com/33839093/108020957-e2d95f80-7060-11eb-8239-9493161381d5.png">

- 단순하게 activation function을 사용하지 않고, 시간적인 흐름을 조절해 memory를 잊어버릴지 유지할지 결정하는 network를 사용한다.
- input, forget, output 3개의 gate를 사용한다.
- LSTM은 hidden state를 계산하는 식이 전통적인 RNN보다 복잡하고, cell state라는 값을 추가한 구조이다. (C<sub>t</sub> = t시점의 셀 상태)
- LSTM은 RNN과 비교해 긴 시퀀스의 입력을 처리하는데 탁월한 성능을 보인다.
- LSTM의 활성화 함수 sigmoid, tanh는 모두 비선형 함수
  - sigmoid함수를 적용하면 0~1사이 값을 가지므로, 기억을 할지(1) 안할지(0) 정해준다.
  -  tanh함수를 적용하면 -1~1사이 값을 가지므로 얼마나 기억할지 정해준다.


### input gate : 현재 정보를 기억하기 위한 게이트
<img src="https://user-images.githubusercontent.com/33839093/108022030-1ae1a200-7063-11eb-9835-e279793c2729.png">

- h<sub>t-1</sub>과 x<sub>t</sub>을 받아 시그모이드를 취하고, 같은 값으로 tanh를 취해준 다음 Hadamard product연산을 한 값
- i<sub>t</sub> 범위는 0~1, g<sub>t</sub>의 범위는 -1~1 이기 때문에 각각 강도와 방향을 의미함
- i<sub>t</sub> = h<sub>t-1</sub>과 x<sub>t</sub>을 받아 시그모이드를 취한 값으로, 현재정보를 저장할지 말지를 의미함.
- g<sub>t</sub> = h<sub>t-1</sub>과 x<sub>t</sub>을 받아 하이퍼볼릭탄젠트를 취한 값으로, 현재정보를 얼마나 더할지, scale역할을 함.


### forget gate : 과거 정보를 잊기 위한 게이트
<img src="https://user-images.githubusercontent.com/33839093/108021260-7f036680-7061-11eb-9f93-2b5ed2f1f00e.png">

- h<sub>t-1</sub>과 x<sub>t</sub>값이 들어와 시그모이드를 취한 값 f<sub>t</sub> = 현재정보를 저장할지 말지를 의미함.
- 시그모이드 함수의 출력값은 0~1사이이기 때문에 값이 0에 가까울수록 정보가 많이 삭제됨, 1에 가까울수록 정보를 온전히 기억한다.

### cell state(장기 상태)
<img src="https://user-images.githubusercontent.com/33839093/108021271-8591de00-7061-11eb-9359-c890682058b6.png">

C<sub>t</sub>는 셀 상태로 장기 상태를 의미한다.
입력 게이트에서 구한 i<sub>t</sub>, g<sub>t</sub>값의 원소별 곱(entrywise product)를 구해, 삭제 게이트의 결과값(f<sub>t</sub> x C<sub>t-1</sub>)에 더한다.
- f<sub>t</sub> = 0이면, 현재 시점의 셀 상태를 결정하기 위한 C<sub>t-1</sub>의 영향력은 0이 되어 -> 입력 게이트의 결과만 C<sub>t</sub>에 영향을 미침.
- 반대로 i<sub>t</sub> = 0이면, C<sub>t</sub>값은 C<sub>t-1</sub>에만 의존함 -> 입력 게이트는 완전히 닫고 삭제 게이트만 연 상태

👉 삭제 게이트: 이전 시점의 입력을 얼마나 반영할지 의미함

👉 입력 게이트: 현재 시점의 입력을 얼마나 반영할지 결정함


### output gate and hidden state(단기 상태)
<img src="https://user-images.githubusercontent.com/33839093/108021273-86c30b00-7061-11eb-93f5-e9cb8e910e1c.png">

출력 게이트는 x<sub>t</sub>(현재시점 t의 x값)와 h<sub>t-1</sub>(이전시점 t-1의 은닉상태)이 시그모이드 함수를 지난 값이다.

👉 h<sub>t</sub>(현재 시점 t의 은닉상태)를 구하는데 사용된다.


### LSTM 실습
