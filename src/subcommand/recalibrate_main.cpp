// recalibrate_main.cpp: mapping quality recalibration for GAM files

#include <omp.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>

#include <string>
#include <sstream>

#include <subcommand.hpp>

#include "../alignment.hpp"
#include "../vg.hpp"
#include "../annotation.hpp"
#include "../bag_of_words.hpp"
#include "../model.hpp"

#include <vowpalwabbit/vw.h>
#include <jansson.h>
#include <mxnet-cpp/MxNetCpp.h>

using namespace std;
using namespace vg;
using namespace vg::subcommand;

void help_recalibrate(char** argv) {
    cerr << "usage: " << argv[0] << " recalibrate [options] --model learned.model mapped.gam > recalibrated.gam" << endl
         << "       " << argv[0] << " recalibrate [options] --model learned.model --train compared.gam" << endl
         << endl
         << "options:" << endl
         << "    -T, --train              read the input GAM file, and use the mapped_correctly flags from vg gamcompare to train a model" << endl
         << "    -m, --model FILE         load/save the model to/from the given file" << endl
         << "    -t, --threads N          number of threads to use" << endl
         << "    -b  --bow                bag of words as features" << endl
         << "    -e  --mems               add mems as features" << endl
         << "    -s  --memstats           add mems stats as features" << endl
         << "    -o  --nomapq             remove mapping quality features" << endl;
}

vector<string> parseMems(const char* json){

    vector<string> s;
    string buf = "";

    json_t *json_content;
    json_error_t error;

    json_content = json_loads(json, 0, &error);


    for(int i = 0; i < json_array_size(json_content); i++)
    {
        json_t *nested_array;
        string sequence;

        nested_array = json_array_get(json_content, i);
        
        sequence = json_string_value(json_array_get(nested_array, 0));

        s.push_back(sequence);
    }
    return s;

}

int get_x_counts(string seq, char a){
    int c = 0;
    for(int i = 0; i < seq.length(); i++ ){
        if(seq[i] == a ){
            c++;
        }
    }
    return c;
}

string parseMemStats(const char* json, string read_sequence){

    stringstream s;
    int max_mems = 0;
    int min_mems = std::numeric_limits<int>::max();
    int max_positions_counts = 0;
    int total_mems = 0;
    int c_count = 0;
    int g_count = 0;
    int a_count = 0;
    int t_count = 0;
    int sequence_len = read_sequence.length();

    int read_c = get_x_counts(read_sequence, 'C');
    int read_g = get_x_counts(read_sequence, 'G');

    float read_gc_content = (read_c + read_g)/float(sequence_len);

    json_t *json_content;
    json_error_t error;

    json_content = json_loads(json, 0, &error);

    total_mems = json_array_size(json_content);

    for(int i = 0; i < json_array_size(json_content); i++)
    {
        json_t *nested_array;
        string sequence;

        nested_array = json_array_get(json_content, i);
        
        sequence = json_string_value(json_array_get(nested_array, 0));

        c_count += get_x_counts(sequence, 'C');
        g_count += get_x_counts(sequence, 'G');
        c_count += get_x_counts(sequence, 'T');
        g_count += get_x_counts(sequence, 'A');
        max_positions_counts = json_array_size(json_array_get(nested_array, 1));

        if(sequence.length() > max_mems){
            max_mems = sequence.length();
        }

        if(sequence.length() < min_mems){
            min_mems = sequence.length();
        }

    }

    float gc_content = 1;

    if(g_count + c_count + a_count + t_count != 0){
        gc_content = (g_count + c_count)/float(g_count + c_count + a_count + t_count);
    }



    s << "minMems:"  << min_mems/float(sequence_len)<< " maxMems:"  << max_mems/float(sequence_len)
     << " maxPositionsCounts:" << max_positions_counts << " TotalMems:" << total_mems
     << " GCcontentRation: "  << read_gc_content/gc_content;

    return s.str(); 

}
/// Turn an Alignment into a Vowpal Wabbit format example line.
/// If train is true, give it a label so that VW will train on it.
/// If train is false, do not label the data.
string alignment_to_example_string(const Alignment& aln, bool train, bool bow, bool mems, bool memstats, bool nomapq) {
    // We will dump it to a string stream
    stringstream s;
    
    if (train) {
        // First is the class; 1 for correct or -1 for wrong
        s << (aln.correctly_mapped() ? "1 " : "-1 ");
    }
    
    // Drop all the features into the mepty-string namespace
    s << "| ";
    
    if(!nomapq){
        // Original MAPQ is a feature
        s << "origMapq:" << to_string(aln.mapping_quality()) << " ";
        
        // As is score
        s << "score:" << to_string(aln.score()) << " ";
        
        // And the top secondary alignment score
        double secondary_score = 0;
        if (aln.secondary_score_size() > 0) {
            secondary_score = aln.secondary_score(0);
        }
        s << "secondaryScore:" << to_string(secondary_score) << " ";
        
        // Count the secondary alignments
        s << "secondaryCount:" << aln.secondary_score_size() << " ";
        
        // Also do the identity
        s << "identity:" << aln.identity() << " ";
    }
    
    
    // Bag of words as features
    if(mems && bow){

        vector<string> mems_list = parseMems(get_annotation<string>(aln, "mems").c_str());

        map<string, int> bw = sequence_to_bag_of_words(aln.sequence(), 10);

        for(auto v : mems_list){
            bw = add_sequence_to_bw(bw, v, 10);
        }

        s << bag_of_word_to_string(bw)  << " ";
    }else if(bow){

        s << bag_of_word_to_string(sequence_to_bag_of_words(aln.sequence(), 10))  << " ";
    }else if(mems){
        
        map<string, int> bw;
        vector<string> mems_list = parseMems(get_annotation<string>(aln, "mems").c_str());

        for(auto v : mems_list){
            bw = add_sequence_to_bw(bw, v, 10);
        }
        s << bag_of_word_to_string(bw) << " ";
    }

    if(memstats){
        s << parseMemStats(get_annotation<string>(aln, "mems").c_str(), aln.sequence()) << " " ;
    }


    return s.str();
}

int append_data(const Alignment& aln, vector<mx_float> &X, vector<mx_float> &y, bool train, bool bow, bool mems, bool memstats, bool nomapq){
    int feature_size = 0;
    int kmer = 10;

    if (train) {
        // First is the class; 1 for correct or -1 for wrong
        mx_float correct_class = 0.0;
        //mx_float incorrect_class = 0.0;
        if(aln.correctly_mapped()){
            correct_class = 1.0;
        }else{
            //incorrect_class = 1.0;
            correct_class = 0.0;
        };

        y.push_back(correct_class);
        //y.push_back(incorrect_class);
    }
    
    

    if(!nomapq){
        // Original MAPQ is a feature
        X.push_back(static_cast<float>(aln.mapping_quality()));
        feature_size++;
        // As is score
        X.push_back(static_cast<float>(aln.score()));
        feature_size++;
        // And the top secondary alignment score
        double secondary_score = 0;
        if (aln.secondary_score_size() > 0) {
            secondary_score = aln.secondary_score(0);
        }
        X.push_back(static_cast<float>(secondary_score));
        feature_size++;
        // Count the secondary alignments
        X.push_back(static_cast<float>(aln.secondary_score_size()));
        feature_size++;
        // Also do the identity
        X.push_back(static_cast<float>(aln.identity()));
        feature_size++;
    }

     // Bag of words as features
    if(mems && bow){
        
        vector<string> mems_list = parseMems(get_annotation<string>(aln, "mems").c_str());

        map<string, int> bw = sequence_to_bag_of_words(aln.sequence(), kmer);

        for(auto v : mems_list){
            bw = add_sequence_to_bw(bw, v, kmer);
        }

        bag_of_word_to_float_vec(bw, X);
        feature_size += pow(4, kmer);
    }else if(bow){

        bag_of_word_to_float_vec(sequence_to_bag_of_words(aln.sequence(), kmer), X);
        feature_size += pow(4, kmer);
    }else if(mems){
        
        map<string, int> bw = generate_dict(kmer);
        vector<string> mems_list = parseMems(get_annotation<string>(aln, "mems").c_str());

        for(auto v : mems_list){
            bw = add_sequence_to_bw(bw, v, kmer);
        }

        bag_of_word_to_float_vec(bw, X);
        feature_size += pow(4, kmer);
    }

    return feature_size;
}

int main_recalibrate(int argc, char** argv) {

    if (argc == 2) {
        help_recalibrate(argv);
        exit(1);
    }

    int threads = 1;
    bool train = false;
    string model_filename;
    bool bow = false;
    bool mems = false;
    bool memstat = false;
    bool nomapq = false;
    int c;
    bool nn_predictor = false;
    optind = 2;
    while (true) {
        static struct option long_options[] =
        {
            {"help", no_argument, 0, 'h'},
            {"train", no_argument, 0, 'T'},
            {"bow", no_argument, 0, 'b'},
            {"mems", no_argument, 0, 'e'},
            {"memstats", no_argument, 0, 's'},
            {"nomapq", no_argument, 0, 'o'},
            {"model", required_argument, 0, 'm'},
            {"threads", required_argument, 0, 't'},
            {"neuralnet", no_argument, 0, 'k'}, 
            {0, 0, 0, 0}
        };

        int option_index = 0;
        c = getopt_long (argc, argv, "kosebhTm:t:",
                         long_options, &option_index);

        // Detect the end of the options.
        if (c == -1) break;

        switch (c)
        {

        case 'T':
            train = true;
            break;
            
        case 'm':
            model_filename = optarg;
            break;

        case 't':
            threads = atoi(optarg);
            omp_set_num_threads(threads);
            break;
        case 'b':
            bow = true;
            break;
        case 'e':
            mems = true;
            break;
        case 's':
            memstat = true;
            break;
        case 'o':
            nomapq = true;
            break;
        case 'k':
            nn_predictor = true;
            break;
        case 'h':
        case '?':
            help_recalibrate(argv);
            exit(1);
            break;

        default:
            abort ();
        }
    }

    // With the GAM input

    get_input_file(optind, argc, argv, [&](istream& gam_stream) {
        
        
        // Neural Net layer architecture
        vector<int> layers{8, 8, 8, 4 ,1};
        int output_size = 1;
        Context ctx = Context::cpu();
        NeuralNet nn(layers);
        vector<mx_float> X;
        vector<mx_float> y;
        vector<mx_float> X_t;
        vector<mx_float> y_t; 
        int counter = 0;
        int batch_size = 256;
        int epochs = 100;
        int correct = 0;
        int input_size = 0;
        //NDArray y_pred;
        
        
        if (train) {
            // We want to train a model.
            
            // Get a VW model.
            // Most of the parameters are passed as a command-line option string.
            // We must always pass --no_stdin because
            // <https://github.com/JohnLangford/vowpal_wabbit/commit/7d5754e168c679ff8968f18a967ccd11a2ba1e80>
            // says so.
            string vw_args = "--no_stdin";
            // We need the logistic stuff to make the predictor predict probabilities
            vw_args += " --link=logistic --loss_function=logistic";
            // We need this to do quadradic interaction features (kernel)
            vw_args += " -q ::";
            // Add L2 regularization
            vw_args += " --l2 0.000001";
            
            
            // We also apparently have to decide now what file name we want output to go to and use -f to send it there.
            if (!model_filename.empty()) {
                // Save to the given model
                vw_args += " -f " + model_filename;
                
#ifdef debug
                // Also dump a human-readable version where feature names aren't hashed.
                vw_args += " --invert_hash " + model_filename + ".inv";
#endif
                
            }
            
            // TODO: what do any of the other parameters do?
            // TODO: Note that vw defines a VW namespace but dumps its vw type into the global namespace.
            LogisticReg log_reg(vw_args);

            
            
            function<void(Alignment&)> train_on = [&](Alignment& aln) {
                
                // Turn each Alignment into a VW-format string
                
                if(nn_predictor){
                    input_size = append_data(aln, X, y, train, bow, mems, memstat, nomapq);
                    counter++;


                    if(counter == batch_size){
                        NDArray X_batch = NDArray(Shape(counter, input_size ), ctx, false);
                        X_batch.SyncCopyFromCPU(X.data(), counter * input_size );
                        cerr <<"writing X" << endl;
                        NDArray y_batch = NDArray(Shape(counter, output_size), ctx, false);;
                        y_batch.SyncCopyFromCPU(y.data(), counter * output_size);
                        cerr <<"writing Y" << endl;
                        X.clear();
                        y.clear();
                        nn.set_epoch(epochs);
                        nn.fit(X_batch, y_batch, "adam", input_size, output_size, batch_size, ctx);
                        counter = 0;
                    }
                }else{
                    string example_string = alignment_to_example_string(aln, true, bow, mems, memstat, nomapq);
                    log_reg.learn_example(example_string);
                }
                // Load an example for each Alignment.
                // You can apparently only have so many examples at a time because they live in a ring buffer of unspecified size.
                // TODO: There are non-string-parsing-based ways to do this too.
                // TODO: Why link against vw at all if we're just going to shuffle strings around? We could pipe to it.
                // TODO: vw alo dumps "example" into the global namespace...
                //example* example = VW::read_example(*model, example_string);

                // Now we call the learn method, defined in vowpalwabbit/global_data.h.
                // It's not clear what it does but it is probably training.
                // If we didn't label the data, this would just predict instead.
                //model->learn(example);
                
                // Clean up the example
                //VW::finish_example(*model, example);
                
            };
            
            // TODO: We have to go in serial because vw isn't thread safe I think.
            stream::for_each(gam_stream, train_on);

            if(counter != 0 && nn_predictor){
                NDArray X_batch = NDArray(Shape(counter, input_size), ctx, false);
                X_batch.SyncCopyFromCPU(X.data(), counter * input_size);
                //cerr <<"writing X" << endl;
                NDArray y_batch = NDArray(Shape(counter, output_size), ctx, false);;
                y_batch.SyncCopyFromCPU(y.data(), counter * output_size);
                //cerr <<"writing Y" << endl;
                X.clear();
                y.clear();
                nn.set_epoch(epochs);
                nn.fit(X_batch, y_batch, "adam", 5, 2, counter, ctx);
                counter = 0;
            }
            
            if(nn_predictor){
                cerr << "SAVING MODEL...." << endl;
                nn.save_model(model_filename);
            }
            
            
            // Now we want to output the model.
            // TODO: We had to specify that already. I think it is magic?
            
            // Clean up the VW model
            //VW::finish(*model);
            
        } else {

            string vw_args = "--no_stdin";
            vw* model;
            // We are in run mode
            if(nn_predictor){
                nn.load_model(model_filename);
                //cerr << "Model Loaded" << endl;
            }else{
                if (!model_filename.empty()) {
                    // Load from the given model
                    vw_args += " -i " + model_filename;

                } 

                model = VW::initialize(vw_args);
            }

            //LogisticReg log_reg(vw_args);
            
            // Make the model
            
            
            // Define a buffer for alignments to print
            vector<Alignment> buf;
            
            // Specify how to recalibrate an alignment
            function<void(Alignment&)> recalibrate = [&](Alignment& aln) {
                
                double prob;

                if(nn_predictor){
                    counter++;
                    input_size = append_data(aln, X_t, y_t, true, bow, mems, memstat, nomapq);
                    NDArray X_batch(Shape(1, input_size), ctx, false);
                    X_batch.SyncCopyFromCPU(X_t.data(), 1 * 5);
                    NDArray y_pred(Shape(1, output_size), ctx, false);
                    
                    y_pred = nn.predict(X_batch, ctx);
                    prob = min(max(static_cast<double>(y_pred.At(0, 0)), 0.0), 1.0);
                    //cerr << "before if else" << endl;
                    //cerr << "after if else" << endl;
                    X_t.clear();
                    y_t.clear();
                }else{
                    // Turn each Alignment into a VW-format string
                    string example_string = alignment_to_example_string(aln, false, bow, mems, memstat, nomapq);

                    //prob = log_reg.predict(example_string);
                }
                
                // Load an example for each Alignment.
                //example* example = VW::read_example(*model, example_string);
                
                // Now we call the learn method, defined in vowpalwabbit/global_data.h.
                // It's not clear what it does but it is probably training.
                // If we didn't label the data, this would just predict instead.
                //model->learn(example);
                
                // Get the correctness prediction from -1 to 1
                //double prob = example->pred.prob;
                
                // Convert into a real MAPQ estimate.
                double guess = prob_to_phred(1.0 - prob);
                // Clamp to 0 to 60
                double clamped = max(0.0, min(60.0, guess));
               
#ifdef debug
                cerr << example_string << " -> " << prob << " -> " << guess << " -> " << clamped << endl;
#endif
                
                // Set the MAPQ to output.
                aln.set_mapping_quality(clamped);
                
                // Clean up the example
                //VW::finish_example(*model, example);
                
                
                
                
#pragma omp critical (buf)
                {
                    // Save to the buffer
                    buf.push_back(aln);
                    if (buf.size() > 1000) {
                        // And output if buffer is full
                        write_alignments(std::cout, buf);
                        buf.clear();
                    }
                }
                
                
            };

            // if(counter == 0){
            //     counter = 1.0;
            // }

            //cerr << "ACCURACY : " << double(correct)/counter << endl;
            
            // For each read, recalibrate and buffer and maybe print it.
            // TODO: It would be nice if this could be parallel...
            stream::for_each(gam_stream, recalibrate);
            
            //VW::finish(*model);
            
            // Flush the buffer
            write_alignments(std::cout, buf);
            buf.clear();
            cout.flush();
        }
        
    });

    return 0;
}

// Register subcommand
static Subcommand vg_recalibrate("recalibrate", "recalibrate mapping qualities", main_recalibrate);
