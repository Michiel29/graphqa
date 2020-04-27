# self_inference

Steps to run the code
1. Install latest version of pyarrow and neptune
```console
$ pip install pyarrow==0.16.*
$ pip install neptune-client
$ pip install psutil
```

2. Compile cython files
```console
$ python setup.py build_ext --inplace
```