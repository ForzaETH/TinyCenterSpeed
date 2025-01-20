# TinyCenterSpeed Optimizations for the Nvidia Jetson or Coral TPU

## Convert the Model from PyTorch to TFLite/EdgeTPU

The package requirements are quite delicate and most of the documentation is depricated.
The best way to convert the model is to follow this community-maintained [repo](https://github.com/PINTO0309/onnx2tf).

Follow the instructions in a jupyternotebook in a fresh environment (google colab recommended. TODO: Could be containerized).

A finished colab notebook can be found [here](https://colab.research.google.com/drive/15uAhoY2QU4C7gMnQ3wsF_kgiKIH3B1om?usp=sharing).

*Environment Setup*
```ipynb

!sudo add-apt-repository -y ppa:deadsnakes/ppa
!sudo apt-get -y update
!sudo apt-get -y install python3.9
!sudo apt-get -y install python3.9-dev
!sudo apt-get -y install python3-pip
!sudo apt-get -y install python3.9-distutils
!wget https://github.com/PINTO0309/onnx2tf/releases/download/1.7.3/flatc.tar.gz \
  && tar -zxvf flatc.tar.gz \
  && sudo chmod +x flatc \
  && sudo mv flatc /usr/bin/
!python3.9 -m pip install -U setuptools \
  && python3.9 -m pip install -U pip \
  && python3.9 -m pip install -U distlib
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
!python3.9 -m pip install tensorflow==2.12.0 \
  && python3.9 -m pip install -U onnx \
  && python3.9 -m pip install -U nvidia-pyindex \
  && python3.9 -m pip install -U onnx-graphsurgeon \
  && python3.9 -m pip install -U onnxruntime==1.13.1 \
  && python3.9 -m pip install -U onnxsim \
  && python3.9 -m pip install -U simple_onnx_processing_tools \
  && python3.9 -m pip install -U onnx2tf \
  && python3.9 -m pip install -U protobuf==3.20.3 \
  && python3.9 -m pip install -U h5py==3.7.0
!pip install psutil
```

*Convert your model*
```ipynb
!onnx2tf -i <path_to_onnx_model>
```

*Compile for EdgeTPU*
```ipynb
!echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
!sudo apt-get update
!sudo apt-get install edgetpu-compiler
```

```ipynb
!edgetpu_compiler saved_model/<name_of_tf_model>.tflite
```

## Python Dependencies

*Install dependencies*
```sh
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

sudo apt-get install python3-pycoral
sudo apt-get install libedgetpu1-std # Or libedgetpu1-max but says might get very hot
```


## USB Setup

Check if the device is available:

```sh
lsusb
```

Look for an entry that looks something like this:
```sh
Bus 001 Device 002: ID 1a6e:089a Global Unichip Corp.
```

Check the device permissions (with the correct numbers (001/002...) from lsusb):
```sh
ls -l /dev/bus/usb/001/002
```

The output will probably look like this:
```sh
crw-rw-r-- 1 root root 189, 1 Oct 29 12:34 /dev/bus/usb/001/002
```
Which means that only the root user has access to the device.

Check if udev rules exist for the coral:
```sh
ls /etc/udev/rules.d/ | grep coral
```

If no rule exists, create one with:
```sh
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1a6e", ATTR{idProduct}=="089a", MODE="0666"' | sudo tee /etc/udev/rules.d/99-coral.rules
```

and reload it with:
```sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Now unplug and replug the coral and check the device permissions again.
Now root and plugdev should have access.

Add your user to the plugdev group:
```sh
sudo usermod -aG plugdev $USER
```

Verify if your user is printed here:
```
groups
```

## Code usage:

Adapted example from [Coral Docs.](https://coral.ai/docs/edgetpu/tflite-python/#inferencing-example)

```python
import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'mobilenet_v2_1.0_224_quant_edgetpu.tflite')
label_file = os.path.join(script_dir, 'imagenet_labels.txt')
image_file = os.path.join(script_dir, 'parrot.jpg')

#List available Coral devices:
print("Available Edge TPUs:", edgetpu.list_edge_tpus())

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(label_file)
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
  ```

## Nvidia TensorRt
![trt workflow](../images/trt.png)

CenterSpeed can be compiled and optimized for inference on a Nvidia GPU with TensorRT (TRT).
To build the model a Pytorch model is first converted into ONNX format and then compiled for TRT.
The onnx model and the TRT engine can be built in __onnx_and_trt.ipynb__.
