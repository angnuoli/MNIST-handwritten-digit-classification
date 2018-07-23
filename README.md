# Introduction

This project is to classify MNIST handwritten digits using pytorch. Four models are used here and they are in `net.py`

```bash
simpleNet
Activation_Net(add ReLU layer)
Batch_Net
CNN(four conv layers, two max-pooling)
```

# Result

## CNN

```bash
epoch 1
*******
[300/469] Loss: 0.13246, Acc: 0.964453
epoch 2
*******
[300/469] Loss: 0.13163, Acc: 0.976354
epoch 3
*******
[300/469] Loss: 0.07662, Acc: 0.981224
epoch 4
*******
[300/469] Loss: 0.08398, Acc: 0.984844
epoch 5
*******
[300/469] Loss: 0.08622, Acc: 0.986172
epoch 6
*******
[300/469] Loss: 0.03386, Acc: 0.988594
epoch 7
*******
[300/469] Loss: 0.03211, Acc: 0.989349
epoch 8
*******
[300/469] Loss: 0.02646, Acc: 0.990833
epoch 9
*******
[300/469] Loss: 0.01633, Acc: 0.991615
epoch 10
*******
[300/469] Loss: 0.01745, Acc: 0.993073
epoch 11
*******
[300/469] Loss: 0.01913, Acc: 0.993620
epoch 12
*******
[300/469] Loss: 0.01401, Acc: 0.994245
epoch 13
*******
[300/469] Loss: 0.03939, Acc: 0.994635
epoch 14
*******
[300/469] Loss: 0.00753, Acc: 0.994818
epoch 15
*******
[300/469] Loss: 0.00434, Acc: 0.995625
epoch 16
*******
[300/469] Loss: 0.00266, Acc: 0.995625
epoch 17
*******
[300/469] Loss: 0.00850, Acc: 0.996354
epoch 18
*******
[300/469] Loss: 0.02561, Acc: 0.996745
epoch 19
*******
[300/469] Loss: 0.01547, Acc: 0.997266
Test Loss: 0.025485, Acc: 0.991500
```