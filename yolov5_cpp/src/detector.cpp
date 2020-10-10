//
// Created by zbl on 2020/9/23.
//

#include <utils.h>
#include "detector.h"


Detector::Detector(const std::string &model_path, const torch::DeviceType &device_type) : device_(device_type) {
  try {
    module_ = torch::jit::load(model_path);
  } catch (const c10::Error& e) {
    std::cerr << "Load Model Failed!\n" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  half_ = (device_ != torch::kCPU);
  module_.to(device_);

  if (half_) {
    module_.to(torch::kHalf);
  }

  module_.eval();
}


std::vector<std::tuple<cv::Rect, float, int>>
Detector::Run(const cv::Mat& img, float conf_threshold, float iou_threshold) {
  torch::NoGradGuard noGrad;
  std::cout << "----------- start detector -----------" << std::endl;

  /*** 数据预处理 ***/
  auto startTime = std::chrono::high_resolution_clock::now();

  /*** 保持原始比例 ***/
  cv::Mat imgInput = img.clone();
  std::vector<float> padInfo = LetterboxImage(imgInput, imgInput, cv::Size(640, 640));
  const float pad_w = padInfo[0];
  const float pad_h = padInfo[1];
  const float scale = padInfo[2];

  /*** 图片数据预处理 ***/
  cv::cvtColor(imgInput, imgInput, cv::COLOR_BGR2RGB); // BGR -> RGB
  imgInput.convertTo(imgInput, CV_32FC3, 1.0f / 255.0f);   // normalization 1/255
  // 转 tensor
  auto tensorImg = torch::from_blob(imgInput.data, {1, imgInput.rows, imgInput.cols, imgInput.channels()}).to(device_);
  tensorImg = tensorImg.permute({0, 3, 1, 2}).contiguous(); // BHWC -> BCHW(Batch, Channel, Height, Width)

  if (half_) {
    tensorImg = tensorImg.to(torch::kHalf);
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(tensorImg);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto useTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  std::cout << "数据预处理耗时: " << useTime.count() << "ms" << std::endl;

  /*** 推理 ***/
  startTime = std::chrono::high_resolution_clock::now();
  torch::jit::IValue output = module_.forward(inputs);
  endTime = std::chrono::high_resolution_clock::now();
  useTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  std::cout << "推理耗时: " << useTime.count() << "ms" << std::endl;

  /*** 结果解析 ***/
  startTime = std::chrono::high_resolution_clock::now();
  auto detections = output.toTuple()->elements()[0].toTensor();

  // 返回结果模板: batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
  auto result = PostProcessing(detections, conf_threshold, iou_threshold);

  // Note - only the first image in the batch will be used for demo
  auto idxMask = result * (result.select(1, 0) == 0).to(torch::kFloat32).unsqueeze(1);
  auto idxMaskIndex = torch::nonzero(idxMask.select(1, 1)).squeeze();
  const auto& resultDataDemo = result.index_select(0, idxMaskIndex).slice(1, 1, 7);

  // use accessor to access tensor elements efficiently
  const auto& demoData = resultDataDemo.accessor<float, 2>();

  // remap to original image and list bounding boxes for debugging purpose
  std::vector<std::tuple<cv::Rect, float, int>> demoDataVec;
  for (int i = 0; i < result.size(0); i++) {
    auto x1 = static_cast<int>((demoData[i][Det::tl_x] - pad_w)/scale);
    auto y1 = static_cast<int>((demoData[i][Det::tl_y] - pad_h)/scale);
    auto x2 = static_cast<int>((demoData[i][Det::br_x] - pad_w)/scale);
    auto y2 = static_cast<int>((demoData[i][Det::br_y] - pad_h)/scale);
    cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
    std::tuple<cv::Rect, float, int> t = std::make_tuple(rect, demoData[i][Det::score], demoData[i][Det::class_idx]);
    demoDataVec.emplace_back(t);
  }

  endTime = std::chrono::high_resolution_clock::now();
  useTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  // It should be known that it takes longer time at first time
  std::cout << "结果解析 : " << useTime.count() << " ms" << std::endl;

  return demoDataVec;
}


std::vector<float> Detector::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
  auto in_h = static_cast<float>(src.rows);
  auto in_w = static_cast<float>(src.cols);
  float out_h = out_size.height;
  float out_w = out_size.width;

  float scale = std::min(out_w / in_w, out_h / in_h);

  int mid_h = static_cast<int>(in_h * scale);
  int mid_w = static_cast<int>(in_w * scale);

  cv::resize(src, dst, cv::Size(mid_w, mid_h));

  int top = (static_cast<int>(out_h) - mid_h) / 2;
  int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
  int left = (static_cast<int>(out_w) - mid_w) / 2;
  int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

  cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  std::vector<float> padInfo{static_cast<float>(left), static_cast<float>(top), scale};

  return padInfo;
}


torch::Tensor Detector::PostProcessing(const torch::Tensor &detections, float conf_thres, float iou_thres) {
  constexpr int itemAttrSize = 5;
  int batchSize = detections.size(0);
  auto numClasses = detections.size(2) - itemAttrSize; // 80 for coco dataset

  /*** 置信度筛选 ***/
  auto confMask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

  /*** compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5] ***/
  detections.slice(2, itemAttrSize, itemAttrSize + numClasses) *= detections.select(2, 4).unsqueeze(2);

  // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
  torch::Tensor box = torch::zeros(detections.sizes(), detections.options());
  box.select(2, Det::tl_x) = detections.select(2, 0) - detections.select(2, 2).div(2);
  box.select(2, Det::tl_y) = detections.select(2, 1) - detections.select(2, 3).div(2);
  box.select(2, Det::br_x) = detections.select(2, 0) + detections.select(2, 2).div(2);
  box.select(2, Det::br_y) = detections.select(2, 1) + detections.select(2, 3).div(2);
  detections.slice(2, 0, 4) = box.slice(2, 0, 4);

  bool isInitialized = false;
  torch::Tensor output = torch::zeros({0, 7});
  for (int batch_i = 0; batch_i < batchSize; batch_i++) {
    auto det = torch::masked_select(detections[batch_i], confMask[batch_i]).view({-1, numClasses + itemAttrSize});

    if (det.size(0) == 0) { continue; }

    // get the max classes score at each result
    std::tuple<torch::Tensor, torch::Tensor> maxClasses = torch::max(det.slice(1, itemAttrSize, itemAttrSize + numClasses), 1);
    // class score
    auto maxConfScore = std::get<0>(maxClasses);
    // index
    auto maxConfIndex = std::get<1>(maxClasses);

    maxConfScore = maxConfScore.to(torch::kFloat32).unsqueeze(1);
    maxConfIndex = maxConfIndex.to(torch::kFloat32).unsqueeze(1);

    // shape n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
    det = torch::cat({det.slice(1, 0, 4), maxConfScore, maxConfIndex}, 1);

    // get unique classes
    std::vector<torch::Tensor> imgClasses;

    auto len = det.size(0);
    for (int i = 0; i < len; i++) {
      bool found = false;
      for (const auto& cls : imgClasses) {
        auto ret = (det[i][Det::class_idx] == cls);
        if (torch::nonzero(ret).size(0) > 0) {
          found = true;
          break;
        }
      }
      if (!found) {
        imgClasses.emplace_back(det[i][Det::class_idx]);
      }
    }
    // iterating all unique classes
    for (const auto& cls : imgClasses) {
      auto clsMask = det * (det.select(1, Det::class_idx) == cls).to(torch::kFloat32).unsqueeze(1);
      auto classMaskIndex = torch::nonzero(clsMask.select(1, Det::score)).squeeze();
      auto bboxByClass = det.index_select(0, classMaskIndex).view({-1, 6});

      // sort by confidence (desc)
      std::tuple<torch::Tensor,torch::Tensor> sortRet = torch::sort(bboxByClass.select(1, 4), -1, true);
      auto confSortIndex = std::get<1>(sortRet);

      bboxByClass = bboxByClass.index_select(0, confSortIndex.squeeze()).cpu();
      int numByClass = bboxByClass.size(0);

      //  Non-Maximum Suppression (NMS 非极大抑制)
      for (int i=0; i < numByClass - 1; i++) {
        auto iou = GetBoundingBoxIoU(bboxByClass[i].unsqueeze(0), bboxByClass.slice(0, i+1, numByClass));
        auto iouMask = (iou < iou_thres).to(torch::kFloat32).unsqueeze(1);

        bboxByClass.slice(0, i+1, numByClass) *= iouMask;

        // remove from list
        auto nonZeroIndex = torch::nonzero(bboxByClass.select(1, 4)).squeeze();
        bboxByClass = bboxByClass.index_select(0, nonZeroIndex).view({-1, 6});
        // update remain number of detections
        numByClass = bboxByClass.size(0);
      }


      torch::Tensor batchIndex  = torch::zeros({bboxByClass.size(0), 1}).fill_(batch_i);

      if (!isInitialized) {
        output = torch::cat({batchIndex, bboxByClass}, 1);
        isInitialized = true;
      } else {
        auto out = torch::cat({batchIndex, bboxByClass}, 1);
        output = torch::cat({output, out}, 0);
      }
    }
  }

  return output;
}


// return the IoU of bounding boxes
torch::Tensor Detector::GetBoundingBoxIoU(const torch::Tensor &box1, const torch::Tensor &box2) {
  // get the coordinates of bounding boxes
  const torch::Tensor& b1_x1 = box1.select(1, 0);
  const torch::Tensor& b1_y1 = box1.select(1, 1);
  const torch::Tensor& b1_x2 = box1.select(1, 2);
  const torch::Tensor& b1_y2 = box1.select(1, 3);

  const torch::Tensor& b2_x1 = box2.select(1, 0);
  const torch::Tensor& b2_y1 = box2.select(1, 1);
  const torch::Tensor& b2_x2 = box2.select(1, 2);
  const torch::Tensor& b2_y2 = box2.select(1, 3);

  // get the coordinates of the intersection rectangle
  torch::Tensor inter_rect_x1 = torch::max(b1_x1, b2_x1);
  torch::Tensor inter_rect_y1 = torch::max(b1_y1, b2_y1);
  torch::Tensor inter_rect_x2 = torch::min(b1_x2, b2_x2);
  torch::Tensor inter_rect_y2 = torch::min(b1_y2, b2_y2);

  // calc inter area
  torch::Tensor interArea = torch::max(inter_rect_x2 - inter_rect_x1 + 1, torch::zeros(inter_rect_x2.sizes()))
                            * torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));

  // calc union area
  torch::Tensor b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
  torch::Tensor b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

  // calc IoU
  torch::Tensor iou = interArea / (b1_area + b2_area - interArea);

  return iou;
}
