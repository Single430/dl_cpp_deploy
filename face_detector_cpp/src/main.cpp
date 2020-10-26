#include <iostream>

#include "cxxopts.hpp"
#include "FaceDetector.h"
#include "Timer.h"

int main(int argc, const char* argv[]) {
  cxxopts::Options parser(argv[0], "Mobile0.25 or Slim or RFB Implementation For Face Detector System");

  std::cout << "欢迎使用人脸检测系统!" << std::endl;

  parser.allow_unrecognised_options().add_options()
      ("img", "need test img path", cxxopts::value<std::string>()->default_value("../sample.jpg"))
      ("param", "ncnn model param file path", cxxopts::value<std::string>()->default_value("../model/mobile0.25.param"))
      ("bin", "ncnn model bin file path", cxxopts::value<std::string>()->default_value("../model/mobile0.25.bin"))
      ("network", "network in mobile0.25 or slim or rfb", cxxopts::value<std::string>()->default_value("mobile0.25"))
      ("video", "video", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print usage");

  auto opt = parser.parse(argc, argv);
  if (opt.count("help")) {
    std::cout << parser.help() << std::endl;
    exit(0);
  }

  std::string imgPath = opt["img"].as<std::string>();

  // 模型文件
  std::string modelParam = opt["param"].as<std::string>();
  std::string modelBin = opt["bin"].as<std::string>();

  const int maxSide = 320;

  //slim or rfb or retinaface
  bool flag;
  flag = opt["network"].as<std::string>() == "mobile0.25";
  Detector detector(modelParam, modelBin, flag);


  Timer timer;

  cv::Mat img = cv::imread(imgPath);
  if (img.empty()) {
    std::cout << "读取图片失败" << std::endl;
    return -1;
  }
  // scale
  float longSide = std::max(img.cols, img.rows);
  float scale = maxSide/longSide;
  cv::Mat imgScale;
  cv::resize(img, imgScale, cv::Size(img.cols * scale, img.rows * scale));

  std::vector<bbox> boxes;

  timer.start();

  detector.Detect(imgScale, boxes);
  timer.end("识别完成...");
  timer.start();

  // draw image
  for (long unsigned int j=0; j<boxes.size(); ++j) {
    cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
    cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
    char label[80];
    sprintf(label, "%.4f", boxes[j].s);

    cv::putText(img, label, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
    cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
    cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
    cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
    cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
    cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
  }
  cv::imwrite("test.png", img);

  return 0;
}