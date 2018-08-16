#include "bag_of_words.hpp"

//namespace vg {

	using namespace std;

	map<string, int> generate_dict(int kmer){
		map<string, int> bw;
		string letters = "ACGT";
		int dict_size = pow(4, kmer);
		string s;
		int l;
		int val;

		for(int i = 0; i < dict_size; i++){
			val = i;
			s ="";
			for(int k = 0; k < kmer; k++){
				l = val % 4;
				val /= 4;
				s += letters[l];
			}
			bw[s] =0;
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

	map<string, int> sequence_to_bag_of_words(string seq, int kmer){
	    map<string, int> bw = generate_dict(kmer);
	    map<string, int>::iterator it;
	    int val;
	    for(int i = 0; i < seq.length() - (kmer-1); i++){
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

	void bag_of_word_to_float_vec(map<string, int> bw, vector<mx_float> &X){
		
		for(map<string,int>::iterator i = bw.begin(); i != bw.end(); i++){
	        X.push_back(i->second);
	    }
	}
//}