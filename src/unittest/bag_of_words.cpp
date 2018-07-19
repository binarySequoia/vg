/// \file bag_of_words.cpp
///  
/// Unit tests for the bag of words
///

#include <iostream>
#include <string>
#include "catch.hpp"
#include "../bag_of_words.hpp"


namespace vg {
    namespace unittest {
        using namespace std;

        TEST_CASE("Bag of words") {
    
            string test = "AABB";
            map<string, int> ans;
            ans["AA"] = 1;
            ans["AB"] = 1;
            ans["BB"] = 1;

            REQUIRE(bag_of_word_to_string(ans) == "AA:1 AB:1 BB:1 ");

            map<string, int> t = sequence_to_bag_of_words(test, 2);

            REQUIRE(bag_of_word_to_string(ans) == bag_of_word_to_string(t));
        
        }
   
    }
}
        
