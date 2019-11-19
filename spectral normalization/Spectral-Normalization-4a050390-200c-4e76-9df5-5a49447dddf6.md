# Spectral Normalization

논문 : Spectral Normalization for Generative Adversarial Networks

저자 : Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida

# Introduce

GAN에서 훈련을 방해하는 요인은 고차원에서의 density ratio estimation이 어렵고, 이론적으로는 optimal한 Discriminator를 가정하기 때문에 훈련이 쉽지 않다.

network를 훈련할 때, Discriminator가 unstable하기 때문에 GAN에서의 훈련을 보장할 수 없다. 또한
데이터 distribution(=A)의 supp 와 Generate된 distibution(=B)의 supp 이 겹치는 것이 없다면, 
A와 B를 완벽하게 구분할 수 있는 Discriminator가 존재한다는 뜻이다.

최악의 상황에선, Discriminator에서 derivate가 0이 되어버리므로 Generator의 학습이 멈춰버린다.

그래서 이 논문에서는 새로운 weight normalization 을 제시한다.

# Method


<img src="https://latex.codecogs.com/svg.latex?\;D^*_G(x)={q_{data}(x)\over{q_{data}+p_G(x)}}=sigmoid(f^*(x))" title = "\Large&space;D^*_G(x) = {q_{data}(x) \over q_{data}+p_G(x)}"/>
<img src="https://latex.codecogs.com/svg.latex?\;where,{f^*(x)}=\log{q_{data}}(x)-\log{p_G(x)}" title = "\Large&space;D^*_G(x) = {q_{data}(x) \over q_{data}+p_G(x)}"/>


<img src="https://latex.codecogs.com/svg.latex?\;\bigtriangledown_xf^*(x)={1\over{q_{data}(x)}}\bigtriangledown_xq_{data}(x)−{1\over{p_{G}(x)}}\bigtriangledown_xp_G(x)" title = "33" />

여기서 derivate가 unbounded하고 incomputable 할 수 있다. 그래서 우리는 derivate를 제한하기 위해 시도할 것이다.

<img src="https://latex.codecogs.com/svg.latex?\;\|f\|_{Lip}=\sup_{x}\sup_{h\neq{0}}{|f(x+h)-f(x)|\over|h|}=M">
<img src="https://latex.codecogs.com/svg.latex?\;\Rightarrow\,the\,smallest\,value\,M\,such\,that\,\frac{\|f(x)-f(x')\|}{\|x-x'\|}<M,for\,any\,x,x'">

즉, Lipschitz constant가 존재한다면 derivate는 Lipschitz constant보다 작거나 같기 때문에 derivate를 제한할 수 있다.

## Spectral Normalization

<img src="https://latex.codecogs.com/svg.latex?\;\|g\|_{Lip}=\sup_h\sigma(\nabla{g(h)}),\\where\,\sigma(A)\,is\,spectral\,norm\,of\,matrix\,A\,(L2\,matrix\,norm\,of\,A)">

<img src="https://latex.codecogs.com/svg.latex?\;\|f\|_{Lip}\le\|\bold{h}_L{\rightarrow}W^{L+1}\bold{h}_L\|_{Lip}\|a_L\|_{Lip}\|\bold{h}_{L-1}{\rightarrow}W^L\bold{h}_{L-1}\|_{Lip}\dots">
<img src="https://latex.codecogs.com/svg.latex?\;\|a_1\|_{Lip}\|\bold{h}_0{\rightarrow}W^1\bold{h}_0\|_{Lip}=\prod_{l=1}^{L+1}\|(\bold{h}_{l-1}{\rightarrow}W^l\bold{h}_l-1)\|_{Lip}=\prod_{l=1}^{L+1}\sigma(W^l).">
<img src="https://latex.codecogs.com/svg.latex?\;\bar{W}_{SN}(W):=W/\sigma(W)">

모든 W 들을 W의 spectral norm이 1을 넘지 못하도록 만들어주는 것을 Spectral Normalization이다. 
다시 말해, 모든 Weight들을 W의 spectral norm 으로 나누어서 f의 Lipschitz constant을 1보다 작게 만들어 derivate에 대한 constraint를 설정한다.


여기서 위에서 계산되는 gradient는 Normalization을 하지 않아도 계산되는 gradient이고
           밑에서 계산되는 gradient는 Normalization을 해서 추가적으로 계산되는 gradient이다.

## Loss Function

<img src="https://latex.codecogs.com/svg.latex?\;V_D(\hat{G},D)=\mathbb{E}_{x{\sim}q_{data}}[\min(0,-1+D(x))]+\mathbb{E}_{z{\sim}p(z)}[\min(0,-1-D(\hat{G}(z)))]">

<img src="https://latex.codecogs.com/svg.latex?\;V_G(G,\hat{D})=-\mathbb{E}_{z{\sim}p(z)}[\hat{D}(G(z))]">

Loss Function 에서는 hinge loss 를 사용했다. 이것이 FID(Frechet Inception Distance)를 올려주는 퍼포먼스를 보여주기 좋다.

<img src="https://latex.codecogs.com/svg.latex?\;-\lambda\mathbb{E}_{\hat{x}{\sim}p_{\hat{x}}}[(\|\bigtriangledown_{\hat{x}}D({\hat{x}})\|2-1)^2]">

또한, WGAN-GP 에서 제시됐던 regularzation term에 Gradient penalty 를 사용한다. 이것은 Resnet기반으로 하는 GAN 에서 Impressive한 퍼포먼스를 보여준다고 한다. ( 논문 피셜 )
