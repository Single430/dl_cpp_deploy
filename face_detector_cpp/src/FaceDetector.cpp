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
    if (*(ptr1+1) > _threshold) {
      box tmp = anchor[i];
      box tmp1;
      bbox result;

      //location and conf
      tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
      tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
      tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
      tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

      result.x1 = (tmp1.cx - tmp1.sx/2) * in.w;
      if (result.x1<0)
        result.x1 = 0;
      result.y1 = (tmp1.cy - tmp1.sy/2) * in.h;
      if (result.y1<0)
        result.y1 = 0;
      result.x2 = (tmp1.cx + tmp1.sx/2) * in.w;
      if (result.x2>in.w)
        result.x2 = in.w;
      result.y2 = (tmp1.cy + tmp1.sy/2)* in.h;
      if (result.y2>in.h)
        result.y2 = in.h;
      result.s = *(ptr1 + 1);

      // landmark
      for (int j = 0; j < 5; ++j)
      {
        result.point[j]._x =( tmp.cx + *(landms + (j<<1)) * 0.1 * tmp.sx ) * in.w;
        result.point[j]._y =( tmp.cy + *(landms + (j<<1) + 1) * 0.1 * tmp.sy ) * in.h;
      }
      totalBox.push_back(result);
    }
    ptr += 4;
    ptr1 += 2;
    landms += 10;
  }
  std::sort(totalBox.begin(), totalBox.end(), cmp);
  nms(totalBox, _nms);
  printf("识别到的目标数: %d\n", (int)totalBox.size());

  for (size_t j = 0; j < totalBox.size(); ++j)
  {
    boxes.push_back(totalBox[j]);
  }

}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h, bool flag) {
  anchor.clear();
  int n;
  std::vector<float> steps;
  std::vector<std::vector<int>> minsize;
  if (flag) {
    n = 3;
    steps = {8, 16, 32};
    minsize = {{10, 20}, {32, 64}, {128, 256}};
  } else {
    n = 4;
    steps = {8, 16, 32, 64};
    minsize = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
  }
  std::vector<std::vector<int>> feature_map(n), min_sizes(n);
  for (size_t i=0; i<feature_map.size(); ++i) {
    feature_map[i].push_back(ceil(h/steps[i]));
    feature_map[i].push_back(ceil(w/steps[i]));

    min_sizes[i] = minsize[i];
  }

  for (size_t k = 0; k < feature_map.size(); ++k)
  {
    std::vector<int> min_size = min_sizes[k];
    for (int i = 0; i < feature_map[k][0]; ++i)
    {
      for (int j = 0; j < feature_map[k][1]; ++j)
      {
        for (size_t l = 0; l < min_size.size(); ++l)
        {
          float s_kx = min_size[l]*1.0/w;
          float s_ky = min_size[l]*1.0/h;
          float cx = (j + 0.5) * steps[k]/w;
          float cy = (i + 0.5) * steps[k]/h;
          box axil = {cx, cy, s_kx, s_ky};
          anchor.push_back(axil);
        }
      }
    }
  }
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

void Detector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH) {
  std::vector<float>vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i)
  {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
               * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i)
  {
    for (int j = i + 1; j < int(input_boxes.size());)
    {
      float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
      float w = std::max(float(0), xx2 - xx1 + 1);
      float   h = std::max(float(0), yy2 - yy1 + 1);
      float   inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH)
      {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      }
      else
      {
        j++;
      }
    }
  }
}
