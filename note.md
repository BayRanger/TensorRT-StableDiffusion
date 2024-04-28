## Task 1

1. convert torch model to onnx model
   in this step, you would generate four onnx files wrt the four modules

```
python export_onnx.py
```

In this procedure, you should set the input and output data shape of each submodule, namely clip, control net, controlled unet, and decoder. Moreover, you should unify the input and outputs name for further trt conversion. Besides, it is necessary to do the sanity check. Also, you could use fixed shape to export the model.


2. convert onnx model to trt model

```

python onnx2trt.py
```


make sure the input and output names match!

3. benchmark the model


For the trt model

time cost is:  2252.4, Perceptual distance: 0.18


For the torch model

time cost is 3707, and perceptual distance is 0.13
