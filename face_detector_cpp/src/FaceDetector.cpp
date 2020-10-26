//
// Created by zbl on 2020/10/24.
//

#include <algorithm>
#include "Timer.h"
#include "FaceDetector.h"


Detector::Detector():
  _nms(0.4),
  _threshold(0.6),
  _mean_val{104.f, 117.f, 123.f},
  _retinaface(false),
  Net(new ncnn::Net())
{}

inline void Detector::Release() {
  if (Net != nullptr){
    delete Net;
    Net = nullptr;
  }
}

Detector::Detector(const std::string &model_param, const std::string &model_bin, bool retinaface):
  _nms(0.4),
  _threshold(0.6),
  _mean_val{104.f, 117.f, 123.f},
  _retinaface(retinaface),
  Net(new ncnn::Net())
{
  Init(model_param, model_bin);
}

void Detector::Init(const std::string &model_param, const std::string &model_bin) {
  Net->load_param(model_param.c_str());
  Net->load_model(model_bin.c_str());
}

Detector::~Detector() {
  Release();
}

void Detector::Detect(cv::Mat &bgr, std::vector<bbox> &boxes) {
  Timer timer;
  timer.start();

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, bgr.cols, bgr.rows);
  in.substract_mean_normalize(_mean_val, 0);

  timer.end("输入图片简单处理...");
  timer.start();

  ncnn::Extractor ex = Net->create_extractor();
  ex.set_light_mode(true);
  ex.set_num_threads(4);
  ex.input(0, in);

  // loc, class, landmark
  std::vector<ncnn::Mat> outs;
  std::vector<std::string> layers{"output0", "586", "585"};
//  std::vector<std::string> layers{"output0", "530", "529"};

  for (const auto& layer: layers) {
    ncnn::Mat out;
    ex.extract(layer.c_str(), out);
    outs.emplace_back(out);
  }

  timer.end("检测完毕...");
  timer.start();

  std::vector<box> anchor;
  create_anchor(anchor,  bgr.cols, bgr.rows, _retinaface);

  timer.end("创建anchor完毕...");
  timer.start();

  std::vector<bbox> totalBox;
  float *ptr = outs[0].channel(0);
  float *ptr1 = outs[1].channel(0);
  float *landms = outs[2].channel(0);

  for (size_t i=0; i<anchor.size(); ++i) {

  }

}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h, bool flag) {

}


bool Detector::cmp(bbox a, bbox b) {
  return a.s > b.s;
}

void Detector::SetDefaultParams() {
  _nms = 0.4;
  _threshold = 0.6;
  _mean_val[0] = 104;
  _mean_val[1] = 117;
  _mean_val[2] = 123;
  Net = nullptr;
}
