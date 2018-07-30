#ifndef VG_MODEL_HPP
#define VG_MODEL_HPP
#include <iostream>
#include <chrono>
#include "mxnet-cpp/MxNetCpp.h"
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>

using namespace std;
using namespace mxnet::cpp;


class LogisticReg
{
	private:
		string load_data_from_alingment();
	public:
		void fit();
		void predict();
		void predict_proba();

};

class NeuralNet
{
	private:
		struct {
			vector<int> layers;
			int batch_size;
		} net_params;
		struct {
			int input_size = 10;
			int batch_size = 100;
			int max_epoch = 100;
			double clip_gradient = 10;
			double learning_rate = 0.000001;
			double weight_decay = 1e-2;
			double momentum = 0.9;

		} train_params;
		bool model_loaded = false;
		bool data_infered = false;
		Symbol model;
		map<string, NDArray> args_map;
		Symbol build_neural_network(const vector<int> &layer_list); 
		void randomized(NDArray &X_data, NDArray &y_data, Context ctx);
		Optimizer* set_optimizer(string optimizer_selection, size_t batch_size);
	public:
		NeuralNet(const vector<int> &layer_list);
		~NeuralNet();
		void load_model(const string &filename);
		void save_model(string filename);
		void set_epoch(size_t epoch);
		void fit(NDArray X_train, NDArray y_train, 
					string optimizer_selection, 
					size_t input_size, size_t output_size , size_t batch_size, Context ctx);
		NDArray predict(NDArray X_test, Context ctx);
		//void predict_proba();
		map<string, NDArray> set_input_shape(NDArray X_train, NDArray y_train, size_t batch_size, Context ctx);
};
#endif