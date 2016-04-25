---
layout: post
title: 人脸检测算法性能评估、综述及进展
---

* content
{:toc}


标签（空格分隔）： 人脸检测

---

本文是在后面给出的网页的基础上整理出来的，做了一些修改和扩充。

## 1.    性能指标概念介绍
先考虑下面的例子。假定某个班级有100人，其中男生80人，女生20人，目标是找出所有女生。某人挑选出50个人，除了所有女生之外，还把30个男生当成了女生。如何评估他的工作？
　　
　　
首先可以用准确率（Accuracy）作为衡量标准。对于给定的测试数据集，准确率就是分类器正确分类的样本个数在总样本数中所占的比例，也就是损失函数是0-1损失时测试数据集上的准确率。简单来说，前面的例子中有80男生和20女生，而某人（也就是定义中的分类器）的分类结果是50男生和50女生。Accuracy指的是他分正确的人占总人数的比例。容易得到，他把其中70人（20女+50男）判定正确了，而总人数是100人，所以他的Accuracy就是70%。


虽然准确率可以在某种意义上评估一个分类器的性能，但是在很多场合并不适用。举个例子，google的索引中共有10,000,000个页面，其中包含100个某网站的页面。从索引中随机抽一个，判断它是不是某网站的页面。如果一个分类器把所有的页面都判断为“不是某网站的页面”，那么Accuracy可以达到99.999%(9,999,900/10,000,000)，同时这样做的效率非常高(return false，一句话)。显然这个分类器并不实用，那还有什么衡量标准呢？这就是下面要介绍的Precision、Recall和F值(F-Measure)。


在给出定义之前，需要先介绍TP、FN、FP、TN四种分类情况。在前面的例子中，我们要挑选出一个班级中的所有女生，也就是说将女生看作“正类”，男生看作“负类”。
  ![此处输入图片的描述][1]
  
  
通过这张表，可以很容易得到例子中这几个分类的值: TP=20, FP=30, FN=0, TN=50

## 2.	指标定义
1)	True Negative Rate（TNR、真阴性率或特异度specificity）：
TNR = TN /（FP + TN）
2)	False Positive Rate（FPR、假阳性率或误报率）：
FPR = FP /（FP + TN）
3)	False Negative Rate（FNR、假阴性率或漏报率） ：
FNR = FN /（TP + FN）
4)	True Positive Rate（TPR、真阳性率、召回率、查全率、recall或灵敏度（sensitivity）：
R = TP / (TP + FN )
5)	Precision（精确率、查准率、检准率）：
P = TP / (TP + FP )
6)	F-Measure（F值）：
 
定义 F_beta为P和 R 的调和平均数，得到
 
F-Measure又称为F-Score，是IR（信息检索）领域的常用的一个评价标准，计算公式为：其中β 是参数，P是精确率(Precision)，R是召回率(Recall)。
当参数β=1时，就是最常见的F1-Measure了：

“召回率”与“准确率”虽然没有必然的关系（从上面公式中可以看到），然而在大规模数据集合中，这两个指标却是相互制约的。
7)	G-Measure：
 
F-measure是召回率和精确率的调和平均数（harmonic mean），而G-measure是召回率和精确率的（ geometric mean）
8)	PBC（Percentage of bad classification）
 
9)	AP和mAP(mean Average Precision)
mAP是为解决P，R，F-measure的单点值局限性的。为了得到 能够反映全局性能的指标，可以考察下图，图中两条曲线(方块点与圆点)分别对应了两个检索系统的准确率-召回率曲线
 
可以看出，虽然两条性能曲线有所交叠但是以圆点标示的系统的性能在绝大多数情况下要远好于用方块标示的系统。从中可以发现，如果一个系统的性能较好，其曲线应当尽可能的向上突出，也就是说曲线与坐标轴之间的面积应当越大。最理想的系统包含的面积应当是1，而所有系统包含的面积都应当大于0。这就是用以评价信息检索系统的最常用性能指标，平均准确率mAP其规范的定义如下：
 
10)	ROC（Receiver Operating Characteristic）和AUC
ROC关注两个指标：TPR和FPR。在ROC 空间中，每个点的横坐标是FPR，纵坐标是TPR，这也就描绘了分类器在TP（真正的正例）和FP（错误的正例）间的trade-off。ROC的主要分析工具是一个画在ROC空间的曲线——ROC curve。

我们知道，对于二值分类问题，实例的值往往是连续值。通过设定一个阈值可以将实例分类到正类或者负类（比如大于阈值划分为正类）。因此我们可以改变阈值，根据不同的阈值进行分类，根据分类结果计算得到ROC空间中相应的点，连接这些点就形成ROC curve。ROC curve经过（0,0）（1,1），实际上(0, 0)和(1, 1)连线形成的ROC curve实际上代表的是一个随机分类器。一般情况下，这个曲线都应该处于(0, 0)和(1, 1)连线的上方，如下图所示。
 
用ROC curve来表示分类器的performance很直观好用。可是，人们总是希望能有一个数值来标志分类器的好坏。于是Area Under roc Curve(AUC)就出现了。顾名思义，AUC的值就是处于ROC curve下方的那部分面积的大小。通常，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的Performance。

AUC计算工具：
http://mark.goadrich.com/programs/AUC/
P/R和ROC是两个不同的评价指标和计算方式，一般情况下，检索用前者，分类、识别等用后者。


  [1]: http://www.cvrobot.net/wp-content/uploads/2015/06/recall-precision-false-positive-false-negative-1.png



## 3.	人脸测评数据库
1)	Face Detection Data Set and Benchmark（FDDB）
网址：http://vis-www.cs.umass.edu/fddb/results.html
路径：z:\User\team02\Face_DB\FDDB\
FDDB是由马萨诸塞大学计算机系维护的一套公开数据库，为来自全世界的研究者提供一个标准的人脸检测评估平台，其中涵盖在自然环境下的各种姿态的人脸；该校还维护了LFW等知名人脸数据库供研究者做人脸识别的研究。作为全世界最具权威的人脸检测评估平台之一，FDDB使用Faces in the Wild，数据库中的2845张图片（包含5171张人脸）作为测试集，而其公布的评估结果也代表了人脸检测的世界最高水平。

FDDB更新更及时一些，所以本文的资料还是主要参考的FDDB。

2)	Fine-grained evaluation of face detection in the wild（MALF）
网址：http://www.cbsr.ia.ac.cn/faceEvaluation/results.html
路径：z:\User\team02\Face_DB\MALF\
该测试网站是由李子青老师的研究组创立和维护的，其性能评估更细致，分析不同分辨率、角度、性别、年龄等条件下的算法准确率。该测试集更新没有FDDB及时。

3)	WIDER FACE: A Face Detection Benchmark
网址：http//mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html
路径：z:\User\team02\Face_DB\WIDER\
WIDER FACE数据库由香港中文大学的汤晓鸥老师所在的实验室创立和维护，包含32203张图片和393703个在尺度，姿态和遮挡方面有高度差异的人脸。该数据库是按照61个事件整理的，对于每一个事件，分别随机抽取40%、10%、50%的数据作为训练、验证和测试集。类似于MALF，WIDERFACE的测试集没有提供ground-truth bounding boxes，用户需要将检测结果提交到网站获得官方的评估结果。

4)	其他数据库
IJB-A dataset: IJB-A is proposed for face detection and face recognition. IJB-A contains 24,327 images and 49,759 faces.
AFW dataset: AFW dataset is built using Flickr images. It has 205 images with 473 labeled faces. For each face, annotations include a rectangular bounding box, 6 landmarks and the pose angles.

## 4.	人脸检测综述

1)	2010年微软zhang cha和张正友撰写的人脸检测的综述报告
       [MSR-TR-2010] A_survey_of_recent_advances_in_face_detection
       
2)	Stefanos Zafeiriou, Cha Zhang和张正友撰写了最新的人脸检测的综述paper，将出版在2016年的《Computer Vision and Image Understanding》
       [CVIU 2015] A Survey on Face Detection in the wild past, present and future
最新性能总结如下：
 
•	在过去的10年人脸检测的性能已经有了激动人心的提升。
•	这些引人注目的性能提升，主要还是得益于将Viala-Jones的boosting和鲁棒性的特征相组合。
•	始终有15~20%的性能Gap，即使允许一个相对较大的FP(大约1000）,始终有15~10%的人脸无法被检测到。需要特别指出的是这些Gap主要是由于是失焦的人脸（比如模糊的人脸）。
•	在这个Benchmark中，最好的基于boosting技术和最好的基于DPM的技术是比较接近的。当然最好的技术还是boosting和DPM组合在一起的性能。（这个就是指的[ECCV 2014] Joint Cascade Face Detection and Alignment）

## 5.	人脸检测最新进展
1)	Face Detection with a 3D Model.   A. Barbu, N. Lay, G. Gramajo.
2)	A Convolutional Neural Network Cascade for Face Detection. H. Li , Z. Lin , X. Shen, J. Brandt and G. Hua. [CVPR2015]
3)	Multi-view Face Detection Using Deep Convolutional Neural Networks. S. S. Farfade, Md. Saberian and Li-Jia Li. [ICMR 2015]  （yahoo的人脸检测）
4)	Aggregate channel features for multi-view face detection.. B. Yang, J. Yan, Z. Lei and S. Z. Li. [IJCB 2014]
5)	A Method for Object Detection Based on Pixel Intensity Comparisons Organized in Decision Trees. CoRR 2014. N. Markus, M. Frljak, I. S. Pandzic, J. Ahlberg and R. Forchheimer. Code：https://github.com/nenadmarkus/pico
6)	Face detection without bells and whistles. ECCV 2014. M. Mathias, R. Benenson, M. Pedersoli and L. Van Gool.[ECCV 2014]  Code：https://bitbucket.org/rodrigob/doppia
7)	The fastest deformable part model for object detection J. Yan, Z. Lei, L. Wen, S. Z. Li
8)	Joint Cascade Face Detection and Alignment. ECCV 2014. D. Chen, S. Ren, Y. Wei, X. Cao, J. Sun. [ECCV 2014]

## 6.	参考网页
http://www.cvrobot.net/recall-precision-false-positive-false-negative/
http://blog.csdn.net/yechaodechuntian/article/details/37394967
http://www.cvrobot.net/latest-progress-in-face-detection-2015/


