# 🐥 Week 03-Seq2Seq


<img src="https://user-images.githubusercontent.com/33839093/110072191-bd6a8680-7dc0-11eb-8458-e862613c3d6e.JPG" width="600">

> Seq2Seq 이전까지는 규칙&통계기반 방법으로 기계 번역을 수행함

### Sequence to Sequence Model
<img src="https://user-images.githubusercontent.com/33839093/110072264-da06be80-7dc0-11eb-85a4-93f4fc506378.png" width="600">

> - Seq2Seq : 하나의 sequence에서 다른 sequence로 번역을 하겠다, sequence는 일반적으로 하나의 문장을 의미함 (ex. 한국어를 영어로 바꾸는 것)
> 1) 일반적으로 입력 토큰으로 단어를 사용함
> 2) 딥러닝 모델은 문자보다 숫자를 잘 처리하기 때문에 모든 단어(토큰)은 워드 임베딩을 통해 임베딩 벡터로 변환됨
> 3) Encoder : 연속적인 각각의 단어(guten, abend)가 들어가 고정된 크기의 context vector를 만듦
> - 입력 sequence에 대한 정보를 적절히 표현할 수 있는 vector를 반환함
> 4) Decoder : context vector를 번역 대상 나라의 말로 변환함
> - context vector가 bottle neck이 될 수 있지만 당시에는 Seq2Seq로도 성능을 많이 향상시킴
> - Encoder의 마지막 hidden state만을 context vector로 사용함 (encoder와 decoder는 서로 다른 파라미터를 가짐)

<img src="https://user-images.githubusercontent.com/33839093/110072310-ea1e9e00-7dc0-11eb-8d6d-1fd7d0e8ea87.png" width="600">

> - Encoder를 여러 개 거치면서 앞쪽에 있는 단어에 대한 정보는 서서히 작아지는 경우가 많음
> - 입력 문장의 순서를 뒤바꾸면 앞쪽에 있는 단어의 정보를 잘 전달할 수 있음  

<img src="https://user-images.githubusercontent.com/33839093/110072315-ebe86180-7dc0-11eb-9a6a-c7681f6ba63c.png" width="450">

> - RNN 셀은 (t-1)에서의 은닉 상태(hidden state)와 t에서의 입력 벡터를 받아 t에서의 은닉 상태를 만듦
> - 이런 구조에서 현재 시점 t에서의 은닉 상태 = 과거의 동일한 RNN 셀에서 모든 은닉 상태의 값의 누적
> - Context vector = 마지막 RNN 셀의 은닉 상태값 = 입력 문장의 모든 단어 토큰들의 정보를 요약한 벡터

### Teacher Forcing
> - 훈련 시에 사용
> - 보통 RNN에서는 (t-1)번째 출력값을 t번째 입력값으로 넣어 은닉 상태를 얻음
> - 하지만, teacher forcing을 사용하면 (t-1)번째 정답을 t번째 입력값으로 넣어 은닉 상태를 얻음
> - 잘못된 예측값을 입력값으로 넣어 또 다른 예측을 하면, 예측이 계속 틀리는 경우를 만들지 않기 위해 사용함
