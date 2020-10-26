//
// Created by zbl on 2020/10/24.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H


#include <opencv2/opencv.hpp>

#include "net.h"

struct Point {
  float _x;
  float _y;
};

struct box {
  float cx;
  float cy;
  float sx;
  float sy;
};

struct bbox {
  float x1;
  float y1;
  float x2;
  float y2;
  float s;
  Point point[5];
};

class Detector {
public:
  Detector();

  void Init(const std::string &model_param, const std::string &model_bin);

  Detector(const std::string &model_param, const std::string &model_bin, bool retinaface = false);

  inline void Release();

//  void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

  void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);

  void create_anchor(std::vector<box> &anchor, int w, int h, bool flag = false);

  inline void SetDefaultParams();

  static inline bool cmp(bbox a, bbox b);

  ~Detector();

public:
  float _nms;
  float _threshold;
  float _mean_val[3];
  bool _retinaface;

  ncnn::Net *Net;
};

#endif //FACE_DETECTOR_H
