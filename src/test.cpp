#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>  // ✅ For timing
#include "fasttext.h"
#include "vector.h"

void getEmbedding(fasttext::FastText& model, const std::string& text, std::vector<float>& embedding) {
    std::istringstream text_stream(text);

    // ✅ Convert std::vector<float> to fasttext::Vector
    fasttext::Vector fasttext_embedding(model.getDimension()); 

    // ✅ Start timing embedding computation
    auto start_time = std::chrono::high_resolution_clock::now();

    // ✅ Call getSentenceVector() with FastText's Vector
    model.getSentenceVector(text_stream, fasttext_embedding);

    // ✅ End timing embedding computation
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Embedding computation time: " << elapsed.count() << " seconds\n";

    // ✅ Copy FastText's Vector to std::vector<float>
    embedding.assign(fasttext_embedding.data(), fasttext_embedding.data() + model.getDimension());
}

int main() {
    fasttext::FastText model;

    // ✅ Start timing model loading
    auto start_load = std::chrono::high_resolution_clock::now();
    
    model.loadModel("cc.en.300.bin");  // Load FastText model

    // ✅ End timing model loading
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = end_load - start_load;
    std::cout << "Model loading time: " << load_time.count() << " seconds\n";

    std::string sentence = "This is a test sentence.";
    std::vector<float> embedding;

    getEmbedding(model, sentence, embedding);

    std::cout << "Embedding size: " << embedding.size() << "\n";
    for (float val : embedding) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    return 0;
}
