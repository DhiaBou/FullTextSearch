#include <iostream>
#include <string>
#include "fts_engine.hpp"
#include "algorithms/inverted/inverted_index_engine.hpp"
#include "algorithms/trigram/trigram_index_engine.hpp"
#include "algorithms/vsm/vector_space_model_engine.hpp"
#include "documents/document.hpp"
#include "documents/document_iterator.hpp"

int main() {
    std::unique_ptr<FullTextSearchEngine> engine;

    // Specify path to the data 
    std::string directory_path;
    std::cout << "Enter the absolute path to the data: ";
    std::getline(std::cin, directory_path);
    DocumentIterator it(directory_path);

    // Choose the algorithm
    std::string algorithm_choice;
    do {
        std::cout << "Select search algorithm (vsm/inverted/trigram): ";
        std::getline(std::cin, algorithm_choice);
        if (algorithm_choice == "vsm") {
            engine = std::make_unique<VectorSpaceModelEngine>();
        } else if (algorithm_choice == "inverted") {
            engine = std::make_unique<InvertedIndexEngine>();
        } else if (algorithm_choice == "trigram") {
            engine = std::make_unique<TrigramIndexEngine>();
        } else {
            std::cout << "Invalid choice!" << std::endl;
        }
    } while (engine == nullptr);

    // Build the index
    engine->indexDocuments(std::move(it));

    // Search
    std::string query;
    while (true) {
        std::cout << "Enter search query: ";
        std::getline(std::cin, query);
    
        auto results = engine->search(query);

        for (const auto &doc: results) {
            std::cout << "Document ID: " << doc << std::endl;
        }
    }
}
