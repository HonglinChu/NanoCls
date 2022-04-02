#include <stdio.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp> 

#include "platform.h"
#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

cv::Mat resize_image(const cv::Mat& img, int size, bool keep_ratio)
{
	int wsize = size;
	int hsize = size;
	if (keep_ratio)
	{
		int img_w = img.cols;
		int img_h = img.rows;
		if (img_h > img_w)
		{
			hsize = int(img_h * wsize / img_w);
		}
		else
		{
			wsize = int(img_w * hsize / img_h); 
		}
	}
	cv::Mat resize_img; 
	cv::resize(img, resize_img, cv::Size(wsize, hsize));
	return resize_img;
}

cv::Mat center_crop(const cv::Mat& img, const int cropSize)
{
	const int offsetW = (img.cols - cropSize) / 2;
	const int offsetH = (img.rows - cropSize) / 2;
	const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
	cv::Mat crop_img = img(roi);
	return crop_img.clone();
}

static int detect_mobilenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	//
	ncnn::Net mobilenet;
	//
#if NCNN_VULKAN
	mobilenet.opt.use_vulkan_compute = true;
#endif
	//       
	mobilenet.load_param("./model/nanocls_mobilenetv2_garbage_sim.param");
	mobilenet.load_model("./model/nanocls_mobilenetv2_garbage_sim.bin");
	//        
	cv::Mat resize_img = resize_image(bgr, 256, true);
	cv::Mat crop_img = center_crop(resize_img, 224);
	//      
	ncnn::Mat in = ncnn::Mat::from_pixels(crop_img.data, ncnn::Mat::PIXEL_BGR2RGB,
		crop_img.cols, crop_img.rows);
	const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
	const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
	//        
	in.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Extractor ex = mobilenet.create_extractor();
	//  
	ex.input("input", in);
     
	ncnn::Mat out; 
	ex.extract("results", out); 
	{
		ncnn::Layer* softmax = ncnn::create_layer("Softmax");    

		ncnn::ParamDict pd; 
		softmax->load_param(pd);  

		softmax->forward_inplace(out, mobilenet.opt); 

		delete softmax;
	}

	out = out.reshape(out.w * out.h * out.c);

	cls_scores.resize(out.w);
	//      
	for (int j = 0; j < out.w; j++)
	{
		//std::cout << out[j] << std::endl;
		cls_scores[j] = out[j];
	}

	return 0;
}
void rstrip(std::string& s) {
	size_t n = s.find_last_not_of(" \n");
	if (n != std::string::npos)
	{
		s.erase(n + 1, s.size() - n); 
	}
	n = s.find_first_not_of(" \n");
	if (n != std::string::npos)
	{
		s.erase(0, n);
	}
}

std::vector<std::string> get_label_name(const std::string file_path)
{
	// 
	std::ifstream infile;
	// 
	infile.open(file_path.data());
	if (!infile.is_open())
	{
		throw "not open file_path";
	}
	//  
	std::vector<std::string> file_contents;
	//  
	std::string temp;
	while (std::getline(infile, temp))
	{
		//
		rstrip(temp);
		// 
		file_contents.push_back(temp);
	}
	infile.close();
	return file_contents;
}

static int print_topk(const std::vector<float>& cls_scores,
	const std::string file_path, int topk)
{
	// 
	std::vector<std::string> label_infos = get_label_name(file_path);
	// 
	int size = cls_scores.size();
	//   
	std::vector< std::pair<float, std::string> > vec;
	//  
	vec.resize(size);
	//  
	for (int i = 0; i < size; i++)
	{
		vec[i] = std::make_pair(cls_scores[i], label_infos[i]);
	}
	//   
	std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
		std::greater< std::pair<float, std::string> >());

	// 
	for (int i = 0; i < topk; i++)
	{
		float score = vec[i].first;
		std::string label_name = vec[i].second;
		//std::cout << score << std::endl;
		//std::cout << label_name << std::endl;
		std::cout << label_name << ":" << score << std::endl;
	}

	return 0;
}

int main(int argc, char** argv) { 

	//const char* imagepath = argv[1];
	const char* imagepath = "./imgs";
	const std::string file_path = "./labels/synset_words.txt"; 

	std::vector<cv::String> filenames;

	cv::glob(imagepath,filenames, false);

	for (auto img_name : filenames)
	{
		cv::Mat m = cv::imread(img_name, 1);
		if (m.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
			return -1;
		}

		#if NCNN_VULKAN  
			ncnn::create_gpu_instance();
		#endif // NCNN_VULKAN  

		std::vector<float> cls_scores;
		detect_mobilenet(m, cls_scores);

		#if NCNN_VULKAN
			ncnn::destroy_gpu_instance();
		#endif // NCNN_VULKAN

		print_topk(cls_scores, file_path, 2);

	}

	return 0;
}