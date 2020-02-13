# synthetic_mnist_uncertainty

Experiments on uncertainty for synthetic mnist data

### Train

```python
python train.py
```

Output:

```
Train on 60000 samples
Epoch 10/10
60000/60000 - 9s - loss: 0.0996 - accuracy: 0.9828

Train on 60000 samples
Epoch 10/10
60000/60000 - 2s - loss: 0.1826 - accuracy: 0.9493

Train on 60000 samples
Epoch 10/10
60000/60000 - 2s - loss: 0.2255 - accuracy: 0.9483
```

### Predict

```python
python predict.py -m path/to/model.h5 path/to/image.jpg
```

Output:

```
BEWY62d.jpg -> entropy=3.340127e-05 class=9 score=1.000
```