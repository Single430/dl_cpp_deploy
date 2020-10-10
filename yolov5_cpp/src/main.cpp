//
// Created by zbl on 2020/9/23.
//

#include <iostream>
#include <memory>
#include <chrono>

#include "detector.h"
#include "cxxopts.hpp"

#define random(x) (rand()%x)

std::vector<std::string> LoadLabels(const std::string& path) {
  // load labels
  std::vector<std::string> labels;
  std::ifstream infile(path);
  if (infile.is_open()) {
    std::string line;
    while (getline(infile, line)) {
      labels.emplace_back(line);
    }
    infile.close();
  } else {
    std::cerr << "Error Read Labels File!\n" << std::endl;
  }

  return labels;
}

void Demo(cv::Mat& img,
          const std::vector<std::tuple<cv::Rect, float, int>>& data_vec,
          const std::vector<std::string>& class_names,
          bool label = true) {
  for (const auto & data : data_vec) {
    cv::Rect box;
    float score;
    int class_idx;
    int r = random(255);
    int g = random(255);
    int b = random(255);

    std::tie(box, score, class_idx) = data;

    cv::rectangle(img, box, cv::Scalar(r, g, b), 2);

    if (label) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << score;
      std::string s = class_names[class_idx] + " " + ss.str();

      auto font_face = cv::FONT_HERSHEY_DUPLEX;
      auto font_scale = 0.5;
      int thickness = 1;
      int baseline=0;
      auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
      cv::rectangle(img,
                    cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                    cv::Point(box.tl().x + s_size.width, box.tl().y),
                    cv::Scalar(r, g, b), -1);
      cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                  font_face , font_scale, cv::Scalar(0, 0, 0), thickness);
    }
  }

  cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
  cv::imshow("Result", img);
  cv::waitKey(0);
}


int main(int argc, const char* argv[]) {
  cxxopts::Options parser(argv[0], "LibTorch inference implementation for YoloV5.");

  // Parser
  parser.allow_unrecognised_options().add_options()
      ("weights", "model.torchscript.pt path", cxxopts::value<std::string>())
      ("img", "a img path", cxxopts::value<std::string>())
      ("labels", "a labels path", cxxopts::value<std::string>()->default_value("../weights/coco.names"))
      ("conf-thres", "object confidence threshold", cxxopts::value<float>()->default_value("0.5"))
      ("iou-thres", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.7"))
      ("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("false"))
      ("view-img", "display results", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print usage");

  auto opt = parser.parse(argc, argv);
  if (opt.count("help")) {
    std::cout << parser.help() << std::endl;
    exit(0);
  }

  // check gpu flag and set device type - CPU/GPU
  bool is_gpu = opt["gpu"].as<bool>();
  torch::DeviceType deviceType;
  if (torch::cuda::is_available() && is_gpu) {
    deviceType = torch::kCUDA;
  } else {
    deviceType = torch::kCPU;
  }

  // load class names from dataset for visualization
  std::vector<std::string> labels = LoadLabels(opt["labels"].as<std::string>());
  if (labels.empty()) {
    return -1;
  }

  // load input image
  std::string img_path = opt["img"].as<std::string>();
  cv::Mat img = cv::imread(img_path);
  if (img.empty()) {
    std::cerr << "Load img failed!\n" << std::endl;
    return -1;
  }
  std::cout << "导入图片: " << opt["img"].as<std::string>() << std::endl;

  // load model
  std::string model_path = opt["weights"].as<std::string>();
  auto detector = Detector(model_path, deviceType);

//   inference
  float conf_thres = opt["conf-thres"].as<float>();
  float iou_thres = opt["iou-thres"].as<float>();
  auto result = detector.Run(img, conf_thres, iou_thres);

//   visualize detections
  if (opt["view-img"].as<bool>()) {
    Demo(img, result, labels);
  }

}
