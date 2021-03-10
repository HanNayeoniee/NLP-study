# 🐫 Week 04-Attention

### 기존 Seq2Seq 모델
- Seq2seq의 문제점 : 하나의 고정된 크기의 context vector가 소스 문장의 모든 정보를 가지고 있어 성능이 저하됨
- Attention : 매번 소스 문장에서의 출력 전부를 입력으로 받아 context vector를 만들면 성능이 높아지지 않을까?
👉 encoder부분의 모든 state를 참고해 context vector를 만듦

### Seq2Seq with Attention
<img src="https://user-images.githubusercontent.com/33839093/110445680-b665b000-8101-11eb-8741-ba39da83a83a.png" width="600">

> - 각각의 단어를 거치면서 갱신되는 hidden state값들(h1, h2, h3, h4)을 모두 저장해뒀다가 출력문장을 생성할 때 모두 반영하겠다.
> - 에너지 : 소스 문장의 단어 벡터 중 어떤 단어에 더 집중할지 결정함

### Decoder
<img src="https://user-images.githubusercontent.com/33839093/110445840-e1e89a80-8101-11eb-8e23-ae21efd176ed.png" width="600">

> - i = 현재의 디코더가 처리 중인 인덱스
> - j = 각각의 인코더 출력 인덱스
> - decoder는 한번에 하나의 단어를 만든다.

1) 에너지 e_ij (attention score) 구하기
> - 에너지 : 인코더의 어느 입력시간 스텝에 집중할지 점수화(수치화), 디코더가 출력값을 만들 때마다 모든 j를 고려함
> - S : 디코더가 출력한 hidden state
> - h : 인코더 부분 각각의 hidden state
> - 👉 디코더가 이전에 출력한 hidden state(S(i-1))와 인코더의 모든 출력값(h_j)를 통해 에너지 값을 구함
> - 👉 어떤 hidden state(h값)와 가장 연관이 많은지를 에너지 값으로 알 수 있음
> - 👉 attention score를 구하는 [함수](https://wikidocs.net/22893)에는 여러 종류가 있음, [Luong](https://hcnoh.github.io/2019-01-01-luong-attention), [Bahdanau](https://hcnoh.github.io/2018-12-11-bahdanau-attention)가 유명함

2) 가중치 alpha_ij (attention distribution) 구하기
> - 가중치 : 에너지값에 softmax를 취해 얻은 확률값으로 0-1사이의 값을 가지며, 어떤 hidden state와 연관성이 높은지 알 수 있음

3) context vector Ci (attention output) 구하기
> - 가중치(alpha_ij)가 반영된 hidden state값으로, 입력 sequence의 weighted sum
> - 위의 그림에서 a_t,1 a_t,2 a_t,3 a_t,T에 해당함
> - 👉 가중치가 반영된 인코더값의 출력값을 구할 수 있음

4) decoder hidden state Si 구하기
<img src="https://user-images.githubusercontent.com/33839093/110446407-7a7f1a80-8102-11eb-8b5b-06afdf5f73cf.png" width="350">

> - S는 decoder의 hidden state, Y는 decoder의 output에 해당함
> - Attention을 적용하면 decoder가 매번 단어를 예측?할 때마다 context vector를 구하기 때문에 C가 아닌 Ci가 사용됨
> - 현재의 hidden state인 Si를 만들기 위해서는 이전에 decoder가 출력한 hidden state값(S(i-1))과 인코더의 모든 hidden state(h1, h2, h3, ...hT)을 묶어서 에너지 값을 구한 뒤에 softmax를 취해 비율값(a_t,1 a_t,2 … a_t.3)을 구할 수 있음
> - 👉 1, 2단계에서 구한 값은 스칼라 값이고 3, 4단계에서 구한 값은 벡터이다.

### 순전파/ forward 과정
<img src="https://user-images.githubusercontent.com/33839093/110447074-350f1d00-8103-11eb-967e-44808f595a31.png" width="450">

> - S1, Y1, C1은 주어진다고 가정
> - Y2, S2를 구하는 식을 보면, Y2 = g(S2, C2), S2 = f(S1, Y1, C2), C2 = 시그마(alpha_2j * h_j)
>  - e_2j -> alpha_2j -> C2 -> S2 -> Y2 순서대로 값을 구할 수 있음

<img src="https://user-images.githubusercontent.com/33839093/110447084-37717700-8103-11eb-8bf6-ab33b8377fd8.png" width="500">

> - 단어(토큰) 4개가 주어진 예제
> - 각 단어는 임베딩을 거쳐 hidden state(h1, h2, h3, h4)값, 벡터 형태임
> - 1. attention score 구하기 : alignment model을 통해 e_21, e_22, e_23, e_24를 구할 수 있음
S1까지 처리했으므로 현재 디코더의 인덱스인 i=2에 해당함
> - 2. attention distribution 구하기 : attention score에 softmax를 적용해 alpha_21, alpha_22, alpha_23, alpha_24를 구할 수 있음
> - 3. attention output 구하기 : 앞에서 구한 가중치 alpha값들과 hidden state(h1, h2, h3, h4)를 곱해 context vector를 구할 수 있음
> - 4. decoder hidden state 구하기
S2 = f(S1, Y1, C2)에 따라 위에서 계산한 context vector인 C2와 주어진 S1, Y1값으로 S2를 구할 수 있음

<img src="https://user-images.githubusercontent.com/33839093/110447085-37717700-8103-11eb-8ac5-be4e3f93d956.png" width="450">

> - 5. Decoder의 output값 구하기
> Y2 = g(S2, C2)에 따라 전 단계에서 계산한 S2와 context vector C2로 Y2값을 구할 수 있음

### Attention 시각화
<img src="https://user-images.githubusercontent.com/33839093/110447661-c8e0e900-8103-11eb-8e56-c13557fde1a2.png" width="650">

> - 영어 -> 프랑스어 번역 결과와 그 때의 attention score를 matrix형태로 나타낸 것으로 그림에서 검은색은 attention score=0, 흰색은 attention score=1을 의미함
> - 대부분의 단어들이 순서대로 1:1 매칭되는것 같지만,
> - 왼쪽 그림의 단어 3개는 서로 대응되고, 오른쪽 그림에서 "the"를 "I"로 번역하기 위해 "man"에 집중한 것을 알 수 있음(프랑스어는 명사의 성별에 따라 관사가 달라지기 때문)
> 👉 일반적인 딥러닝에서는 파라미터가 너무 많아 파라미터를 분석해 딥러닝이 어떻게 예측하는지 알기 어렵지만, attention은 딥러닝 각 단어를 번역할 때 어떤 단어를 얼마나 참고했는지 알 수 있음

<br>

### 추가 내용
	- e_ij에서 a는 무엇인가?
	  => a는 attention score function에 해당하며 다양한 종류가 있음
  	- BLEU score(https://wikidocs.net/31695)


### References
- [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)
- [[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://www.youtube.com/watch?v=AA621UofTUA)
- [튜토리얼로 익히는 머신러닝/딥러닝](http://blog.naver.com/PostView.nhn?blogId=ckdgus1433&logNo=221608376139)
