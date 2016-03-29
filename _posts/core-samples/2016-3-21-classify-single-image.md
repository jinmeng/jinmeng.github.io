---
layout: post
title: Classify a single image using caffe
---

基于caffe官方网站上面的代码做了一些修改, 去掉了处理leveldb的部分, 对lmdb部分代码做了小小的封装. 
开始是想按照http://blog.csdn.net/lingerlanlan/article/details/39400375的方法做, 最终没没能搞定. 


```
//TODO: add code for read image data from lmdb 20160323 19:03
#include <caffe/caffe.hpp>
#define USE_OPENCV
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

using std::cout;
using std::endl;


/* Pair (label, confidence) representing a prediction. */
//typedef std::pair<string, float> Prediction;
typedef std::pair<int, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);   // ,const string& mean_file,     const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);//N = 5

 private:
//  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  //cv::Mat mean_;
  //std::vector<string> labels_;
  std::vector<int> labels_;
};

Classifier::Classifier(const string& model_file, const string& trained_file) 
{
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  //jin 2016-03-21 18:12:27 
  //net_.reset(new Net<float>(model_file, TEST));
  net_.reset(new Net<float>(model_file));
  Caffe::set_phase(Caffe::TEST);  
  
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    
//  /* Load the binaryproto mean file. */
//  SetMean(mean_file);
//
//  /* Load labels. */
//  std::ifstream labels(label_file.c_str());
//  CHECK(labels) << "Unable to open labels file " << label_file;
//  string line;
//  while (std::getline(labels, line))
//    labels_.push_back(string(line));
//
	
	labels_.push_back(0);
	labels_.push_back(1);
	
  Blob<float>* output_layer = net_->output_blobs()[0];     
  CHECK_EQ(labels_.size(), output_layer->channels())
  //CHECK_EQ(2, output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
    std::sort(pairs.begin(), pairs.end(), PairCompare);
  //std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
  {
    result.push_back(pairs[i].second);
 //   std::cout << pairs[i].second << std::endl;
  }
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);
//  cout << "output.size=" << output.size() << endl;
  //printf("%6.2f %6.2f\n");

  //N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}



std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);
  
  //jin: 
//  net_->Forward();
  //const	vector<shared_ptr<Layer<float> > > &layers = net_->layers();
  //net_->ForwardFromTo(0, layers.size()-1);
  
  
  net_->ForwardPrefilled();
	
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];  //TODO 20160322 19:31
  //Blob<float> *output_layer = net_->blob_by_name("prob");
  
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();

  
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
	  //jin 20160323 16:30
	cv::Mat channel(height, width, CV_32FC1, input_data);
    //cv::Mat channel(height, width, CV_8UC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

//jin 20160323 16:32  divide sample_float by 256  
	sample_float = sample_float * 0.00390625; //  1/256
   

//jin 20160321 19:11:00
//  cv::Mat sample_normalized;
//  cv::subtract(sample_float, mean_, sample_normalized);
//
//  /* This operation will write the separate BGR planes directly to the
//   * input layer of the network because it is wrapped by the cv::Mat
//   * objects in input_channels. */
//  cv::split(sample_normalized, *input_channels);
    cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel img.jpg" << std::endl;
             //<< " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
//  string mean_file    = argv[3];
//  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file);

  string file = argv[3];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

//    std::cout << "predictions.size() = " << predictions.size() << std::endl;
  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


```
