/*
 * crnn
*/
#include "CRnnDeploy.h"

using namespace std;

CRnn::CRnn(string& modelPath, string& keyPath) {
  this->m_module = this->loadModule(modelPath);
  this->m_keys = this->readKeys(keyPath);
}

std::vector<std::string> CRnn::readKeys(const std::string& keyPath) {
  ifstream in(keyPath);
  std::ostringstream tmp;
  tmp << in.rdbuf();
  string keys = tmp.str();

  vector<string> words;
  words.push_back(" ");
  int len = keys.length();
  int i = 0;

  while (i < len) {
     assert((keys[i] & 0xF8) <= 0xF0);
     int next = 1;
     if ((keys[i] & 0x80) == 0x00) {
     } else if ((keys[i] & 0xE0) == 0xC0) {
       next = 2;
     } else if ((keys[i] & 0xF0) == 0xE0) {
       next = 3;
     } else if ((keys[i] & 0xF8) == 0xF0) {
       next = 4;
     }
     words.push_back(keys.substr(i, next));
     i += next;
  }

  return words;
}

torch::jit::script::Module CRnn::loadModule(const string& modelPath) {
  torch::jit::script::Module module;
  try {
     module = torch::jit::load(modelPath);
  } catch (const c10::Error& e) {
     cerr << "load module failed!!\n";
  }

  return module;
}

torch::Tensor CRnn::loadImg(string& imgPath, bool isBath) {
  cv::Mat input = cv::imread(imgPath, 0);
  if(!input.data){
     printf("Error: not image data, imgFile input wrong!!");
  }
//  int resize_h = int(input.cols * 28 / input.rows);
  printf("W: %d, H: %d", input.cols, input.rows);
  cv::resize(input, input, cv::Size(28, 28));
  torch::Tensor imgTensor;
  if(isBath){
     imgTensor = torch::from_blob(input.data, {28, 28, 1}, torch::kByte);
     imgTensor = imgTensor.permute({2, 0, 1});
  }
  else {
     imgTensor = torch::from_blob(input.data, {1, 28, 28, 1}, torch::kByte);
     imgTensor = imgTensor.permute({0, 3, 1, 2}); // b,h,w,c -> b,c,h,w
  }

  imgTensor = imgTensor.toType(torch::kFloat);
  imgTensor = imgTensor.div(255);
  imgTensor = imgTensor.sub(0.5);
  imgTensor = imgTensor.div(0.5);

  return imgTensor;
}

void CRnn::inference(torch::Tensor& input) {
  torch::Tensor pred = this->m_module.forward({input}).toTensor();
  vector<int> predChars;
  auto maxRes = pred.max(1, true);
  int maxIdx = get<1>(maxRes).item<float>();
  cout << maxIdx << endl;
}

int main(int argc, const char* argv[]) {
  if (argc < 4) {
     printf("missing input param !\n");
     return -1;
  }
  string modelPath = argv[1];
  string keyPath = argv[2];
  string imgPath = argv[3];

  cout << "Hello, World!" << endl;
  CRnn* crnn = new CRnn(modelPath, keyPath);
  torch::Tensor imgTensor = crnn->loadImg(imgPath);
  cout << imgTensor.sizes() << endl;
  crnn->inference(imgTensor);
  delete crnn;
//  cv::waitKey(20000);
  return 0;
}
