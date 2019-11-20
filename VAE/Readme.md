# Variational AutoEncoder

논문 : Auto-Encoding Variational Bayes
저자 : Diederik P.Kingma ( Amsterdam Univ. ), Max Welling ( Amsterdam Univ. )

# Introduce

Generative하는 모델을 만들고 싶을 때 어떻게 해야할까?
또 Random한 Distribution을 따르는 변수들에 대해서 의미있는 Generate를 하려면 어떻게 해야할까?

예를 들어,

<img src="https://latex.codecogs.com/svg.latex?\;z{\sim}N(0,1)">

를 따르는 변수를 z 라고 가정하자. 그리고 다음을 추정할 것이다.

<img src="https://latex.codecogs.com/svg.latex?\;p_{\theta^{*}}(x)">

그렇게 하기 위해서 

<img src="https://latex.codecogs.com/svg.latex?\;p_{\theta}(x|z)">

를 구해야 할 것이고, 의미있는 값을 도출해 내기 위해서 다음 두개의 값을 구할 것이다.

<img src="https://latex.codecogs.com/svg.latex?\;(prior)\,\,\,p_{\theta}(x)=\int{p_\theta}(z){p_\theta}(x|z)">
<img src="https://latex.codecogs.com/svg.latex?\;(postrior)\,\,p_{\theta}(z|x)=\frac{p_{\theta}(x|z)p_{\theta}(z)}{p_{\theta}(x)}">

그러나 논문에서는 이것들을 구할 때 2가지 문제점을 갖고 있다고 제시하고 있습니다.

1. Intractability
   - We cannot evaluate or differentiate the marginal likelihood
   - The EM algorithm cannot be used
2. A large dataset
   - we have so much data that batch optimization is too costly

그래서 논문에서는 3가지 해결 방안을 제시하고 있습니다.

1. MLE 를 이용할 것이다.
2. z가 주어졌을 때 x를 추정하는 posterior inference 할 것이다.
3. x의 marginal likelihood를 inference 할 것이다.

<img src="https://latex.codecogs.com/svg.latex?\;{\Rightarrow}\tilde{z}{\sim}q_{\phi}(z|x)"> 를 정의해서 z 에 대한 posterior inference 할 것 이다.

### Varational Bound

Entropy = KL + (variational) Lower Bound

<img src="https://latex.codecogs.com/svg.latex?\;{\log}p_{\theta}(x^{(i)})=D_{KL}(q_{\phi}(z{\vert}x^{(i)}){\vert}{\vert}p_{\theta}(z{\vert}x^{(i)}))+\mathcal{L}({\theta},{\phi};x^{(i)})">

<img src="https://latex.codecogs.com/svg.latex?\;에서\,{\theta},{\phi}">에 대한 <img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}({\theta},{\phi};x^{(i)})"> 의 <img src="https://latex.codecogs.com/svg.latex?\;{\delta}">를 구하고 싶다.

그러나 <img src="https://latex.codecogs.com/svg.latex?\;\theta"> 에 대한 delta 는 high variance를 갖는다

### Method

Data point 에 대하여 marginal likelihood 의 lower bound =<img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}({\theta},{\phi};x^{i})">로 정의한다.
<img src="https://latex.codecogs.com/svg.latex?\;\log(p(x)){\geq}ELBO">이므로 marginal likelihood를 maximize 하기 위해
<img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}({\theta},{\phi};x^{i})">를 maximize 한다.
<img src="https://latex.codecogs.com/svg.latex?\;ELBO=-D_{KL}(q_{\phi}(z|x^{(i)})||p_{\theta}(z))+\mathbb{E}_{q_{\phi}(z|x^{(i)})}[{\log}p_{\theta}(x^{(i)}|z)]"> 로 정의할 때

1. <img src="https://latex.codecogs.com/svg.latex?\;\phi"> 에 대한 posterior distribution (Variational Inference)
2. <img src="https://latex.codecogs.com/svg.latex?\;\theta"> 에 대한 prior distribution (Marginal likelihood Maximize)

θ 에 대해서 ELBO를 Maximize 하면 marginal likelihood를 maximize하는 것이고

Φ에 대해서 ELBO를 Maximize 하면 posterior distribution 의 maximize likelihood를 하는 것이다.

loss = <img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}({\phi},{\theta},x)=-\mathbb{E}_{q_{\phi}(z|x)}[{\log}p_{\theta}(x|g_{\phi}(z))]+D_{KL}(q_{\phi}(z|x)\|p_{\theta}(z))=-ELBO">

결국 ELBO 를 maximize 하기 위한 것으로 dual problem 으로 -ELBO를 minimizing 하는 것으로 바꿔 풀 수 있다.

### Reparameter trick

만일 p(z|x) 가 Gaussian Distribution 을 따른다고 가정한다면, sampling이 힘들다. 따라서 Reparameter trick을 사용해서 계산을 좀 더 쉽고 mu와 sigma의 loss 를 통해 Encoder 의 학습이 가능하다.

<img src="https://latex.codecogs.com/svg.latex?\;z{\sim}p(z|x)=N({\mu},{\sigma}^2)"> 이고 <img src="https://latex.codecogs.com/svg.latex?\;{\epsilon}{\sim}N(0,1)"> 일 때,<img src="https://latex.codecogs.com/svg.latex?\;z={\mu}+{\sigma}{\epsilon}"> 로 표현이 가능하다.
