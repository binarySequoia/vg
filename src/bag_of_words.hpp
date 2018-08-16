#ifndef VG_BAG_OF_WORDS_HPP_INCLUDED
#define VG_BAG_OF_WORDS_HPP_INCLUDED

#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>

//namespace vg {

	using namespace std;

	map<string, int> generate_dict(int kmer);

	map<string, int> add_sequence_to_bw(map<string, int> bw, string seq, int kmer);

	string bag_of_word_to_string(map<string, int> bw);

	map<string, int> sequence_to_bag_of_words(string seq, int kmer);

	void bag_of_word_to_float_vec(map<string, int> bw, vector<mx_float> &X);
//}

#endif