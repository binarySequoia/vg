#include "model.hpp"


LogisticReg::LogisticReg(string vw_args)
{
	model = VW::initialize(vw_args);
	return;
}

LogisticReg::~LogisticReg(){
	VW::finish(*model);
}

void LogisticReg::learn_example(string example_string){
	example* example = VW::read_example(*model, example_string);
	model->learn(example);
	VW::finish_example(*model, example);
}

double LogisticReg::predict(string example_string){
	example* example = VW::read_example(*model, example_string);
	model->learn(example);
	double prob = example->pred.prob;
	VW::finish_example(*model, example);
	return prob;
}

NeuralNet::NeuralNet(const vector<int> &layer_list){
	//arams.layers = layers_list;
	model = build_neural_network(layer_list);
	return;
}

NeuralNet::~NeuralNet(){
	MXNotifyShutdown();
	cout << "Neural Network Deleted" << endl;
}

void NeuralNet::randomized(NDArray &X_data, NDArray &y_data, Context ctx){
	size_t data_size = y_data.GetShape().at(0);
	srand(time(0));
	NDArray temp_x;
	NDArray temp_y;

	for(size_t i = 0; i < data_size; i++){

		temp_x = X_data.Slice(i, i+1).Copy(ctx);
		temp_y = y_data.Slice(i, i+1).Copy(ctx);

		int rand_num = rand();
		int rand_idx =  (rand_num) % data_size;

		X_data.Slice(i, i+1) = X_data.Slice(rand_idx, rand_idx+1).Copy(ctx);
		X_data.Slice(rand_idx, rand_idx+1) = temp_x;

		y_data.Slice(i, i+1) = y_data.Slice(rand_idx, rand_idx+1).Copy(ctx);
		y_data.Slice(rand_idx, rand_idx+1) = temp_y;
	}

	return;
};

Optimizer* NeuralNet::set_optimizer(string optimizer_selection, size_t batch_size){
	const float learning_rate = 0.1;
  	const float weight_decay = 1e-2;
	Optimizer* opt = OptimizerRegistry::Find("sgd");
	opt->SetParam("rescale_grad", 1.0/batch_size)
	 ->SetParam("lr", learning_rate)
	 ->SetParam("wd", weight_decay)
	 ->SetParam("clip_gradient", 10);

    return opt;
}

map<string, NDArray> NeuralNet::set_input_shape( NDArray X_train, NDArray y_train, size_t batch_size, Context ctx)
{
	// Inputs to our network
	map<string, NDArray> args;
	args["data"] = X_train.Slice(0, batch_size).Copy(ctx);
	args["data_label"] = y_train.Slice(0, batch_size).Copy(ctx);
	NDArray::WaitAll();

  	// Let MXNet infer shapes other parameters such as weights
  	model.InferArgsMap(ctx, &args, args);
  	data_infered = true;
  	return args;
}

void NeuralNet::fit(NDArray X_train, NDArray y_train, 
					string optimizer_selection, 
					size_t input_size, size_t output_size, size_t batch_size,
					Context ctx)
{
	cout << "Fitting Neural Network" << endl;


  	size_t data_size = y_train.GetShape().at(0);
  	MAE train_logloss;
  	//Accuracy acu_train;
  	NDArray X_batch, y_batch;

  	if(!data_infered){
  		args_map = set_input_shape(X_train, y_train, batch_size, ctx);
  	}

  	//model.InferArgsMap(ctx, &args_map, args_map);
  	// Let MXNet infer shapes other parameters such as weights
  	
  	if(!model_loaded){
	  	//Xavier xavier = Xavier(Xavier::gaussian, Xavier::in, 2.34);
		auto initializer = Uniform(0.1);
		for (auto &arg : args_map) {
			// be careful here, the arg's name must has some specific ends or starts for
			//  * initializer to call
			//xavier(arg.first, &arg.second);
			initializer(arg.first, &arg.second);
		}
	}

	Optimizer* opt = set_optimizer(optimizer_selection, batch_size);

	auto *exec = model.SimpleBind(ctx, args_map);
  	auto arg_names = model.ListArguments();


	for (int iter = 0; iter < train_params.max_epoch; ++iter) {
		LG << "Train Epoch: " << iter;
		/*reset the metric every epoch*/
		train_logloss.Reset();
		randomized(X_train, y_train, ctx);
		auto tic = chrono::system_clock::now();
		/*reset the data iter every epoch*/
		for(size_t i = 0; i < data_size; i += batch_size){

			if(i + batch_size > data_size ){
				continue;
			}else{
				X_batch = X_train.Slice(i, i + batch_size);
				y_batch = y_train.Slice(i, i + batch_size);
			}

      		X_batch.CopyTo(&args_map["data"]);
      		y_batch.CopyTo(&args_map["data_label"]);
      		NDArray::WaitAll();

			exec->Forward(true);
			
			exec->Backward();

			NDArray::WaitAll();
			

			for (size_t j = 0; j < arg_names.size(); ++j) {
				if (arg_names[j] == "data" || arg_names[j] == "data_label"){
					continue;
				}
				opt->Update(j, exec->arg_arrays[j], exec->grad_arrays[j]);
				//cout << exec->grad_arrays[j] << endl;
			}

			//cout << " train_y_batch: "<< y_batch << endl;

			//cout << " train_output_batch: "<< exec->outputs[0] << endl;

			NDArray::WaitAll();
		}

		auto toc = chrono::system_clock::now();

		for(size_t i = 0; i < data_size; i += batch_size){

			if(i + batch_size >= data_size ){
				continue;
			}else{
				X_batch = X_train.Slice(i, i + batch_size);
				y_batch = y_train.Slice(i, i + batch_size);
			}

			X_batch.CopyTo(&args_map["data"]);
      		y_batch.CopyTo(&args_map["data_label"]);
			NDArray::WaitAll();
			exec->Forward(false);
			NDArray::WaitAll();

			//cout << " y_batch: "<< y_batch << endl;

			//cout << " output_batch: "<< exec->outputs[0] << endl;

			train_logloss.Update(y_batch, exec->outputs[0]);
		}

		
		float duration = chrono::duration_cast<chrono::milliseconds>(toc - tic).count() / 1000.0;
		LG << "Epoch: " << iter << " " << data_size/duration
		<< " LogLoss: " << train_logloss.Get();
	}

	delete exec;
	//MXNotifyShutdown();
	return;
}

NDArray NeuralNet::predict(NDArray X_test, Context ctx){
	int data_size = X_test.GetShape().at(0);
  	NDArray output(Shape(data_size, 1), ctx);


  	//if(!data_infered){
  	args_map = set_input_shape(X_test, output, data_size, ctx );
  	auto *exec = model.SimpleBind(ctx, args_map);

  	//}
	X_test.CopyTo(&args_map["data"]);
	
	exec->Forward(false);

	output = exec->outputs[0];
	
	delete exec;

	return output;
}

void NeuralNet::save_model(string filename){
	/*save the parameters*/
    string save_path_param = filename;
    
    auto save_args = args_map;
    /*we do not want to save the data and label*/
    save_args.erase(save_args.find("data"));
    save_args.erase(save_args.find("data_label"));
    /*the alexnet does not get any aux array, so we do not need to save
     * aux_map*/
    LG << " Saving to..." << save_path_param;
    NDArray::Save(save_path_param, save_args);
}

void NeuralNet::load_model(const string& filename){
	ifstream f(filename.c_str());
    if(f.good()){ 
    	LG << " Loading model from ..." << filename;
    	args_map = NDArray::LoadToMap(filename);
    	LG << " Model Loaded";
    	model_loaded = true;
    }else{
    	LG << filename << " not found. unable to load model.";
    }
	
	return;
}

void NeuralNet::set_epoch(size_t epoch){
	train_params.max_epoch = epoch;
}

Symbol NeuralNet::build_neural_network(const vector<int> &layer_list){
	
	auto x = Symbol::Variable("data");
	auto y = Symbol::Variable("data_label");

	vector<Symbol> weights(layer_list.size());
	vector<Symbol> biases(layer_list.size());
	vector<Symbol> outputs(layer_list.size());

	for(size_t i = 0; i < layer_list.size(); i++ ){
		weights[i] = Symbol::Variable("w_"+ to_string(i));
		biases[i] = Symbol::Variable("b_" + to_string(i));
		Symbol fc = FullyConnected((i == 0)? x : outputs[i-1],
									weights[i],
									biases[i],
									layer_list[i]);	// Neuron_num
		outputs[i] = (i == layer_list.size()-1) ? fc : Activation("relu_"+ to_string(i), fc, ActivationActType::kRelu);
	}

	return  SoftmaxOutput(outputs.back(), y);
}


