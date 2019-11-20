# Octave Convolution

논문 : Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution

저자 : Yunpeng Chen, Haoqi Fan, Bing Xu, Zhicheng Yan, Yannis Kalantidis, Marcus Rohrbach, Shuicheng Yan, Jiashi Feng ( Facebook AI )

## Introduce

![](Untitled-2c5fe37e-ac51-4b44-b7fb-ea3dd3eaf766.png)

이 논문의 Motivation 이 되었다고 소개하는 High Frequency, Low Frequency에 관한 내용이다.

H-F는 sharp한 shape를 나타내고, L-F는 좀더 blur한 content를 나타낸다.

이것을 Motivation으로 Convolution을 할 때 receptive field들을 효과적으로 사용하게 할 것이다.

또한, 연산 양이 줄어 속도적인 측면에서도 더 좋은 효과를 기대하게 한다.

## Method

![](_2019-11-19_01-9639f835-f0aa-4c12-bb40-06d72276f71a.37.40.png)

실제로 OctConv 연산을 하기 위해 2개의 input이 필요하고 2개의 output이 나오게 된다.
( 처음 Layer와 마지막 Layer을 제외하고 )

High Frequency를 표현해주는 X^H와 Low Frequency를 표현해주는 X^L가 input으로 들어가야 한다.
결과물로는 High Frequency Y^H, Low Frequency Y^L이 나오게 된다.

$$Y^H = W^{H\rightarrow H}\odot {X^H} + upsample(W^{L\rightarrow H} \odot X^L) \\ Y^L\  = W^{L\rightarrow L} \ \odot X^L \ + \ pooling(W^{H\rightarrow L}\odot X^H)\ \ $$

## Performence

![](Untitled-e235b78f-9558-4b14-bf87-d3f63658bf26.png)

논문에서는 에 의해 감소되는 연산량을 제시하고 있다.

Memory 연산량은 으로 줄었다.

![](Untitled-f6e20ca3-3df9-455c-b50b-e3c3dea87ba8.png)

Memory 연산량은 1-3a/4 으로 줄었다.

# Parameters 는 크게 줄었지만, Testing단계에서의 Accuracy에서는 다른 모델과 큰 차이가 없다.

## Idea

High Frequency는 sharp한 shape를 표현하고, Low Frequency는 blur한 content를 표현한다.

그래서 shape와 content를 나타내는 receptive field를 생성한다는 아이디어를 생각한 것 같다.

즉, H 는 shape에 관여하고 L 은 content에 관여한다고 가정한다면,

Y^L = [Y^H→L + Y^L→L] 에서 Y^H→L은 shape에서의 content부분이고
                                                  Y^L→L은 content에서 content부분이다.

Y^H = [Y^H→H + Y^L→H] 에서 Y^H→H은 shape에서의  shape 부분이고
                                                    Y^L→H은 content에서 shape 부분이다.

최종 output은 shape / content 부분으로 나뉠 것이다. ( 예상 )
