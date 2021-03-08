# 🐫 Week 04-Attention Mechanism

### Seq2seq(encoder-decoder) model
<img src="https://user-images.githubusercontent.com/33839093/110282162-5a2b5f00-8021-11eb-9c4b-d0bd65e7f831.png" width="- 600">

> - 단어별로 번역하는건 별로 좋은 방법이 아님: 예) "I love you"를 한국어로 번역하면 "나는 너를 사랑해"이지만, 이를 단어 기준으로 번역하면 'I-나', 'love-너를', 'you-사랑해'로 번역되는 오류가 있기 때문
> - encoder : 각 단어를 순차적으로 받아 context vector를 만듦
> - decoder : context vector로부터 기계번역을 시작
> - 문제점 : context vector는 고정된 크기의 벡터이기 때문에 문장이 길어지거나 context vector의 크기가 충분하지 않으면 입력 단어의 모든 의미를 함축하기 어려움
> - 해결 방법 : attention mechanism

### Attention Mechanism
<img src="https://user-images.githubusercontent.com/33839093/110282637-2ac92200-8022-11eb-99df-ed670786e24a.png" widt="600">

> - 기존 seq2seq 모델 : encoder에서 나온 맨 마지막 state만 사용해 context vecctor를 구성함
> - encoder에서 나온 state를 decoder에서 dynamic하게 context vector를 만들어서 사용할 수 있도록 하자!
> - 장점 : context vector가 고정된 크기가 아님, encoder의 state에서 집중해야할 단어들만 선택해 집중하는 mechanism을 설계할 수 있음

### Seq2seq with attention mechanism
- step 1
<img src="https://user-images.githubusercontent.com/33839093/110282829-809dca00-8022-11eb-8663-9553adb72128.png" width="600">

> 1) h3 = 전통적인 모델에서의 context vector
> 2) RNN셀의 모든 값을 활용해 FC layer를 거침
> 3) s1, s2, s3는 RNN셀에 있던 encoder의 score를 의미함
> 4) Softmax를 취하면 각각은 확률값을 가짐
	I는 0.9, love는 0.0, you는 0.1의 값을 가짐. 이 값을 attention weight라고 부르며, 각 단어에 얼마나 집중할 것인지 의미함
> 5) Attention weight에 따라 첫번째 context vector를 만듦

- step 2
<img src="https://user-images.githubusercontent.com/33839093/110283097-dffbda00-8022-11eb-9da8-f78ccfc2d047.png" width="600">

> - Decoder의 값인 dh1이 FC layer에 들어감
> - Encoder state값인 h1, h2, h3은 매번 사용됨
> - 바뀐 FC layer(h1, h2, h3, df1)을 통해 attention weight를 다시 계산
> - 두번째 context vector인 cv2를 계산해 모델에 넣어주면 '널'을 출력함

- step 3
<img src="https://user-images.githubusercontent.com/33839093/110283282-27826600-8023-11eb-8910-c7f81deb8426.png" width="600">
> - 3번째 attention weight에서는 'love'의 값이 0.95로 많이 집중하는 것을 볼 수 있으며 '사랑해'를 출력함

### 중요
	- Attention weight를 통해 encoder에서 나온 state중에 어디에 집중할지 결정함
	- Decoding할 때마다 context vector가 달라짐
	- Teacher forcing : 첫번째 decoding 결과가 '난'이 아니라 '너'처럼 틀린 prediction을 하는 경우에는 정답인 '난'을 넣어서 학습시킴

### References
- 논문: Neural machine translation by jointly learning to align and translate
- 딥러닝을 이용한 자연어처리 입문 (https://wikidocs.net/24996)
- 시퀀스 투 시퀀스+어텐션 강의 (https://www.youtube.com/watch?v=WsQLdu2JMgI)
