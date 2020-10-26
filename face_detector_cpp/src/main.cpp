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

  return 0;
}