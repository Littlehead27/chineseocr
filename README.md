## 本项目基于[yolo3](https://github.com/pjreddie/darknet.git) 与[crnn](https://github.com/meijieru/crnn.pytorch.git)  实现中文自然场景文字检测及识别
master分支将保留一周，后续app分支将替换为master 

# 注意的事项 
- python的版本要对 如果出现 安装web.py 版本要对 可能和python3.7 兼容有问题 
'File "C:\anaconda3\lib\site-packages\web\debugerror.py", line 313, in debugerror
    return web._InternalError(djangoerror())
  File "C:\anaconda3\lib\site-packages\web\debugerror.py", line 271, in djangoerror
    _get_lines_from_file(filename, lineno, 7)
  File "C:\anaconda3\lib\site-packages\web\debugerror.py", line 246, in _get_lines_from_file
    source = open(filename).readlines() '
- UnicodeDecodeError: 'gbk' codec can't decode byte 0xa2 in position 452: illegal multibyte sequence
- UnicodeDecodeError: 'gbk' codec can't decode bytes in position 1-2: illegal multibyte sequence．
- 出现以上错误 是编码格式的问题，改一下源代码里面报错的地方。 如'open(filename)'后面加一个，‘，encoding = 'utf-8'’  可解决这个问题

- tensorflow InvalidArgumentError: You must feed a value for placeholder tensor with dtype float 
如果出现这个问题，说明是TensorFlow 1.14版本问题源代码是1.8 的 
# 可将tf改成 1.13.1-gpu 版本
# 如果导入TensorFlow 出错
 -ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory 
 -  这个是说cuda和TensorFlow 版本 不对  TensorFlow1.13.1 不能运行在 cuda 10.1  上  改一下版本 其实TensorFlow1.14 也可以在cuda 10.0 上运行
# 相关版本匹配问题  
## TensorFlow1.13.1 + cuda10.0 +cudnn 7.6 （这个可以运行）
##  tensorflow1.8 + cuda 9.0 （这个可以运行）
## TensorFlow 1.14 +cuda10.0 (会报错 可能要修改代码)
# 实现功能
- [x]  文字方向检测 0、90、180、270度检测（支持dnn/tensorflow） 
- [x]  支持(darknet/opencv dnn /keras)文字检测,支持darknet/keras训练
- [x]  不定长OCR训练(英文、中英文) crnn\dense ocr 识别及训练 ,新增pytorch转keras模型代码(tools/pytorch_to_keras.py)
- [x] 支持darknet 转keras, keras转darknet, pytorch 转keras模型
- [x]  新增对身份证/火车票结构化数据识别
- [ ]  新增语音模型修正OCR识别结果   
- [ ]  新增CNN+ctc模型，支持DNN模块调用OCR，单行图像平均时间为0.02秒以下     
- [ ]  优化CPU调用，识别速度与GPU接近(近期更新)      

## 环境部署

GPU部署 参考:setup.md     
CPU部署 参考:setup-cpu.md   


### 下载编译darknet(如果直接运用opencv dnn或者keras yolo3 可忽略darknet的编译)  
```
git clone https://github.com/pjreddie/darknet.git 
mv darknet chineseocr/
##编译对GPU、cudnn的支持 修改 Makefile
#GPU=1
#CUDNN=1
#OPENCV=0
#OPENMP=0
make 
```

修改 darknet/python/darknet.py line 48    
root = '/root/'##chineseocr所在目录     
lib = CDLL(root+"chineseocr/darknet/libdarknet.so", RTLD_GLOBAL)    


## 下载模型文件   
模型文件地址:
* [baidu pan](https://pan.baidu.com/s/1gTW9gwJR6hlwTuyB6nCkzQ)
* [google drive](https://drive.google.com/drive/folders/1XiT1FLFvokAdwfE9WSUSS1PnZA34WBzy?usp=sharing)

复制文件夹中的所有文件到models目录
   
## 模型转换
pytorch ocr 转keras ocr     
``` Bash
python tools/pytorch_to_keras.py  -weights_path models/ocr-dense.pth -output_path models/ocr-dense-keras.h5
```
darknet 转keras     
``` Bash
python tools/darknet_to_keras.py -cfg_path models/text.cfg -weights_path models/text.weights -output_path models/text.h5
```
keras 转darknet      
``` Bash
python tools/keras_to_darknet.py -cfg_path models/text.cfg -weights_path models/text.h5 -output_path models/text.weights
```

## 编译语言模型
``` Bash
git clone --recursive https://github.com/parlance/ctcdecode.git   
cd ctcdecode   
pip install .  
```
## 下载语言模型  
``` Bash
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
mv zh_giga.no_cna_cmn.prune01244.klm chineseocr/models/
```
## web服务启动
``` Bash
cd chineseocr## 进入chineseocr目录
ipython app.py 8080 ##8080端口号，可以设置任意端口
```

## 构建docker镜像 
``` Bash
##下载Anaconda3 python 环境安装包（https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh） 放置在chineseocr目录下   
##建立镜像   
docker build -t chineseocr .   
##启动服务   
docker run -d -p 8080:8080 chineseocr /root/anaconda3/bin/python app.py

```


## 识别结果展示

<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/train-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/idcard-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/img-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/line-demo.png"/>


## 访问服务
http://127.0.0.1:8080/ocr
# 如果实在服务器上运行的 上面的IP 要换成服务器ip  
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/demo.png"/>


## 参考
1. yolo3 https://github.com/pjreddie/darknet.git   
2. crnn  https://github.com/meijieru/crnn.pytorch.git              
3. ctpn  https://github.com/eragonruan/text-detection-ctpn    
4. CTPN  https://github.com/tianzhi0549/CTPN       
5. keras yolo3 https://github.com/qqwweee/keras-yolo3.git    
6. darknet keras 模型转换参考 参考：https://www.cnblogs.com/shouhuxianjian/p/10567201.html  
7. 语言模型实现 https://github.com/lukhy/masr


# _______________________
一下是本人两个周来复现本项目的心得体会
'''[libprotobuf FATAL google/protobuf/stubs/common.cc:61] This program requires version 3.3.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1. Please update your library. If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library. (Version verification failed in "google/protobuf/descriptor.pb.cc".)
terminate called after throwing an instance of 'google::protobuf::FatalException'
what(): This program requires version 3.3.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1. Please update your library. If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library. (Version verification failed in "google/protobuf/descriptor.pb.cc".)'''

出现以上的错，重新TensorFlow 先 
* pip uninstall tensorboard tensorflow-estimator tensorflow-gpu 
在重新装  pip install tensorflow-gpu==1.13.1  -i https://pypi.tuna.tsinghua.edu.cn/simple/


'''>>> import tensorflow as tf
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/zengyj/.conda/envs/chineseocr/lib/python3.6/site-packages/tensorflow/__init__.py", line 24,
    from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
  File "/home/zengyj/.conda/envs/chineseocr/lib/python3.6/site-packages/tensorflow/python/__init__.py", l
    from tensorflow.core.framework.graph_pb2 import *
  File "/home/zengyj/.conda/envs/chineseocr/lib/python3.6/site-packages/tensorflow/core/framework/graph_p
    from google.protobuf import descriptor as _descriptor
* ImportError: cannot import name 'descriptor'
'''
   
''' Using cached https://pypi.tuna.tsinghua.edu.cn/packages/7b/b1/0ad4ae02e17ddd62109cd54c291e311c4b5fd09b4
* Requirement already satisfied: protobuf>=3.6.1 in ./.conda/envs/chineseocr/lib/python3.6/site-packages (f
*  Requirement already satisfied: six>=1.10.0 in ./.conda/envs/chineseocr/lib/python3.6/site-packages (from
* Requirement already satisfied: gast>=0.2.0 in ./.local/lib/python3.6/site-packages (from tensorflow-gpu==
* Requirement already satisfied: termcolor>=1.1.0 in ./.local/lib/python3.6/site-packages (from tensorflow-
* Requirement already satisfied: keras-applications>=1.0.6 in ./.local/lib/python3.6/site-packages (from te
* Requirement already satisfied: astor>=0.6.0 in ./.local/lib/python3.6/site-packages (from tensorflow-gpu=
* Requirement already satisfied: wheel>=0.26 in ./.conda/envs/chineseocr/lib/python3.6/site-packages (from
* Requirement already satisfied: grpcio>=1.8.6 in ./.local/lib/python3.6/site-packages (from tensorflow-gpu
* Requirement already satisfied: keras-preprocessing>=1.0.5 in ./.local/lib/python3.6/site-packages (from t
* Requirement already satisfied: numpy>=1.13.3 in ./.conda/envs/chineseocr/lib/python3.6/site-packages (fro'''
出现以上的错 
出现这个错一般是protobuf 版本不对
-  pip uninstall protobuf
- pip install protobuf==3.6.1 （TensorFlow 1.13.1 要求这个版本）
- 

