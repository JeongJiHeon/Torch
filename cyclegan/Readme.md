# CycleGAN

논문 : Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

저자 : Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros (UC Berkeley)

## Introduce

이전에 냈던 pix2pix논문을 보면 paired dataset을 Domain으로 하지만, 그러한 dataset을 구하기 쉽지 않고 구하더라도 data의 양이 부족하다.

그런 의미에서 이 CycleGAN은 unpaired dataset을 Domain 으로 설정할 수 있도록 만든 model이라는 것에서 의미가 있다.

pix2pix논문을 다시 한번 살펴보자.

conditionalGAN 에서 label 이 vector가 아니라 Image를 사용했다는 점이 pix2pix의 핵심 중 하나이다.

즉, P( Z = Image | Y = Image ) 를 생각했기 때문에 paired된 dataset이 필수적이다.

그러나 CycleGAN 에서는 label을 사용하지 않고, Reconstruction했을 때 원래의 Domain에 속하도록 하는 Loss를 제시할 것이다.

## Idea

Data를 paired하게 설정한 것이 아니라, Generator의 Architecture가 paired하도록 setting 됐다고 생각해보자. 즉,

<img src="https://latex.codecogs.com/svg.latex?\;x{\in}X,\,\,\,\,\,G(x){\in}Y">
<img src="https://latex.codecogs.com/svg.latex?\;y{\in}Y,\,\,\,\,\,F(y){\in}X">

를 만족하는 Generator G, F가 존재한다고 가정하자. 그리고 우리의 목표는 G와 F 가 inverse 관계가 되도록 만드는 것이다. 따라서

<img src="https://latex.codecogs.com/svg.latex?\;G(F(y)){\approx}y\\F(G(x)){\approx}x">

를 만족해야하고 이것을 새로운 Loss 로써 정의할 것이다 ( Cycle Loss )

## Method

![](Untitled-f285a2bc-a253-4f27-a60b-12c4713fe7f2.png)

1번째 모델을 (Gx, Dx) 으로 정의 하고,
2번째 모델을 (Gy, Dy) 으로 정의 하자.

<img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}_{GAN}(G_y,D_y,X,Y)=\mathbb{E}_{y{\sim}dataY}[{\log}(D_y(y)]+\mathbb{E}_{x{\sim}dataX}[1-{\log}(D_y(G_y(x))]">
<img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}_{GAN}(G_x,D_x,Y,X)=\mathbb{E|_{x{\sim}dataX}[\log(D_x(x)]+\mathbb{E}_{y{\sim}dataY}[1-\log(D_x(G_x(y))]">

기본적인 GAN Loss을 정의하고, reconstruction 을 위한 새로운 Loss를 정의하자.

<img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}_{cyc}(G_x,G_y)=\mathbb{E}_{x{\sim}dataX}[\|G_x(G_y(x))-x\|_1+\mathbb{E}_{y{\sim}dataY}[\|G_y(G_x(y))-y\|_1">

<img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}(G_x,G_y,D_x,D_y)=\mathcal{L}_{GAN}(G_y,D_y)+\mathcal{L}_{GAN}(G_x,D_x)+\lambda\mathcal{L}_{cyc}(G_x,G_y)">

<img src="https://latex.codecogs.com/svg.latex?\;{\therefore}\,\,G_x^*,G_y^*,D_x^*,D_y^*=\arg\min_{G_x,G_y}\max_{D_x,D_y}\mathcal{L}(G_x,G_y,D_x,D_y)">

L1Loss 는 stable한 guide force의 역할을 하고,

Adversarial Loss는 detail 한 부분을 담당한다고 한다.

또한, Identity Loss를 정의했는데

<img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}_{Identity}=\mathbb{E}_{x{\sim}dataX}[\|G_x(x)-x\|_1]+\mathbb{E}_{y{\sim}dataY}[\|G_y(y)-y\|_1]">

CycleGAN 창시자 피셜로는  Identity Loss를 추가했을 경우, 모델이 더 나빠지는 경우는 없었다고 한다.

### Image Pool

'Learning from Simulated and Unsupervised Images through Adversarial Training' 이라는 논문에서 나온 방법이고, unsupervised learning 의 모델을 학습할 때 사용된다.

### Patch GAN

pix2pix논문에서 나온 Discriminator의 방법이고, 이미지를 patch별로 쪼개서 real or label을 확인한다. pixelGAN( patch size 1x1 ) 보다는 patchGAN의 patch의 사이즈에 따라 더 좋은 성능을 보인다고 한다.
