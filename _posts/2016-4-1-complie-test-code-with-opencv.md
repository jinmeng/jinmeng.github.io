---
layout: post
title: Compile test code with opencv
---

之前用opencv仅限于在caffe代码里面调用，直接写个小程序测试才发现编译需要一点技巧，参考http://www.cnblogs.com/woshijpf/p/3840530.html。

加上-L是因为找不到报错：“/usr/bin/ld: cannot find -lcufft”

```
g++ opencv_imshow.cpp -o imshow -L /usr/local/cuda/lib64/ `pkg-config --cflags --libs opencv` 

```



