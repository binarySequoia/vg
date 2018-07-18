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

#include <vowpalwabbit/vw.h>
#include <jansson.h>

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
         << "    -s  --memstats           add mems stats as features" << endl;
}

map<string, int> sequence_to_bag_of_words(string seq, int kmer){
    map<string, int> bw;// = generate_dict(kmer);
    map<string, int>::iterator it;
    int val;
    for(int i = 0; i < seq.length() - kmer; i++){
        string current_seq = seq.substr(i, kmer);
        it = bw.find(current_seq);
        if(it == bw.end()){
            bw[current_seq] = 1;
        }else{
            bw[current_seq] += 1;
        }
    }
    return bw;
}

map<string, int> add_sequence_to_bw(map<string, int> bw, string seq, int kmer){
    map<string, int>::iterator it;
    int val;
    for(int i = 0; i < seq.length() - kmer; i++){
        string current_seq = seq.substr(i, kmer);
        it = bw.find(current_seq);
        if(it == bw.end()){
            bw[current_seq] = 1;
        }else{
            bw[current_seq] += 1;
        }
    }
    return bw;
}

string bag_of_word_to_string(map<string, int> bw){
    stringstream s;
    for(map<string,int>::iterator i = bw.begin(); i != bw.end(); i++){
        s << i->first << ":" << i->second << " ";
    }
    return s.str();
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
    int min_mems = 0xffffffff;
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

    float gc_content = (g_count + c_count)/float(g_count + c_count + a_count + t_count);

    s << "minMems:"  << min_mems/float(sequence_len)<< " maxMems:"  << max_mems/float(sequence_len)
     << " maxPositionsCounts:" << max_positions_counts << " TotalMems:" << total_mems
     << " GCcontentRation: "  << read_gc_content/gc_content;

     cerr << s.str() << endl;
    return s.str(); 

}
/// Turn an Alignment into a Vowpal Wabbit format example line.
/// If train is true, give it a label so that VW will train on it.
/// If train is false, do not label the data.
string alignment_to_example_string(const Alignment& aln, bool train, bool bow, bool mems, bool memstats) {
    // We will dump it to a string stream
    stringstream s;
    
    if (train) {
        // First is the class; 1 for correct or -1 for wrong
        s << (aln.correctly_mapped() ? "1 " : "-1 ");
    }
    
    // Drop all the features into the mepty-string namespace
    s << "| ";
    
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
    
    
    // Bag of words as features
    if(mems && bow){

        vector<string> mems_list = parseMems(get_annotation<string>(aln, "mems").c_str());

        map<string, int> bw = sequence_to_bag_of_words(aln.sequence(), 4);

        for(auto v : mems_list){
            bw = add_sequence_to_bw(bw, v, 4);
        }

        s << bag_of_word_to_string(bw)  << " ";
    }else if(bow){

        s << bag_of_word_to_string(sequence_to_bag_of_words(aln.sequence(), 4))  << " ";
    }else if(mems){
        
        map<string, int> bw;
        vector<string> mems_list = parseMems(get_annotation<string>(aln, "mems").c_str());

        for(auto v : mems_list){
            bw = add_sequence_to_bw(bw, v, 4);
        }
        s << bag_of_word_to_string(bw) << " ";
    }

    if(memstats){
        s << parseMemStats(get_annotation<string>(aln, "mems").c_str(), aln.sequence());
    }


    return s.str();
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
    int c;
    optind = 2;
    while (true) {
        static struct option long_options[] =
        {
            {"help", no_argument, 0, 'h'},
            {"train", no_argument, 0, 'T'},
            {"bow", no_argument, 0, 'b'},
            {"mems", no_argument, 0, 'e'},
            {"memstats", no_argument, 0, 's'},
            {"model", required_argument, 0, 'm'},
            {"threads", required_argument, 0, 't'},
            {0, 0, 0, 0}
        };

        int option_index = 0;
        c = getopt_long (argc, argv, "sebhTm:t:",
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
        case 'h':
        case '?':
            help_recalibrate(argv);
            exit(1);
            break;

        default:
            abort ();
        }
    }

    get_input_file(optind, argc, argv, [&](istream& gam_stream) {
        // With the GAM input
       
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
            vw* model = VW::initialize(vw_args);
            
            function<void(Alignment&)> train_on = [&](Alignment& aln) {
                
                // Turn each Alignment into a VW-format string
                string example_string = alignment_to_example_string(aln, true, bow, mems, memstat);
                
                // Load an example for each Alignment.
                // You can apparently only have so many examples at a time because they live in a ring buffer of unspecified size.
                // TODO: There are non-string-parsing-based ways to do this too.
                // TODO: Why link against vw at all if we're just going to shuffle strings around? We could pipe to it.
                // TODO: vw alo dumps "example" into the global namespace...
                example* example = VW::read_example(*model, example_string);
                
                // Now we call the learn method, defined in vowpalwabbit/global_data.h.
                // It's not clear what it does but it is probably training.
                // If we didn't label the data, this would just predict instead.
                model->learn(example);
                
                // Clean up the example
                VW::finish_example(*model, example);
            };
            
            // TODO: We have to go in serial because vw isn't thread safe I think.
            stream::for_each(gam_stream, train_on);
            
            // Now we want to output the model.
            // TODO: We had to specify that already. I think it is magic?
            
            // Clean up the VW model
            VW::finish(*model);
            
        } else {
            // We are in run mode
            
            string vw_args = "--no_stdin";
            
            if (!model_filename.empty()) {
                // Load from the given model
                vw_args += " -i " + model_filename;
            }
            
            // Make the model
            vw* model = VW::initialize(vw_args);
       
            // Define a buffer for alignments to print
            vector<Alignment> buf;
            
            // Specify how to recalibrate an alignment
            function<void(Alignment&)> recalibrate = [&](Alignment& aln) {
                
                // Turn each Alignment into a VW-format string
                string example_string = alignment_to_example_string(aln, false, bow, mems, memstat);
                
                // Load an example for each Alignment.
                example* example = VW::read_example(*model, example_string);
                
                // Now we call the learn method, defined in vowpalwabbit/global_data.h.
                // It's not clear what it does but it is probably training.
                // If we didn't label the data, this would just predict instead.
                model->learn(example);
                
                // Get the correctness prediction from -1 to 1
                double prob = example->pred.prob;
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
                VW::finish_example(*model, example);
                
                
                
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
            
            // For each read, recalibrate and buffer and maybe print it.
            // TODO: It would be nice if this could be parallel...
            stream::for_each(gam_stream, recalibrate);
            
            VW::finish(*model);
            
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
