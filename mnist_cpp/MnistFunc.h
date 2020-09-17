//
// Created by zbl on 2020/7/29.
//

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <cassert>
#include <vector>

#ifndef CRNN_H
#define CRNN_H

class CRnn{
  public:
    CRnn(std::string& modelPath, std::string& keyPath);
    static torch::Tensor loadImg(std::string& imgPath, bool isBath=false);
    void inference(torch::Tensor& input);

  private:
    torch::jit::script::Module m_module;
    std::vector<std::string> m_keys;
    std::vector<std::string> readKeys(const std::string& keyPath);
    torch::jit::script::Module loadModule(const std::string& modelPath);
};

#endif //CRNN_H
