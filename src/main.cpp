#include <iostream>
#include <string>
#include "FullTextSearchEngine.hpp"
#include "algorithms/InvertedIndex/InvertedIndexEngine.hpp"
#include "algorithms/TrigramIndex/TrigramIndexEngine.hpp"
#include "algorithms/VectorSpaceModel/VectorSpaceModelEngine.hpp"
#include "dataUtils/Document.hpp"
#include "dataUtils/DocumentIterator.hpp"
#include "dataUtils/DocumentUtils.hpp"

int main() {
    std::unique_ptr<FullTextSearchEngine> engine;

    // Specify path to the data 
    std::string directoryPath;
    std::cout << "Enter the absolute path to the data: ";
    std::getline(std::cin, directoryPath);
    DocumentIterator it(directoryPath);

    // Choose the algorithm
    std::string algorithmChoice;
    do {
        std::cout << "Select search algorithm (vsm/inverted/trigram): ";
        std::getline(std::cin, algorithmChoice);
        if (algorithmChoice == "vsm") {
            engine = std::make_unique<VectorSpaceModelEngine>();
        } else if (algorithmChoice == "inverted") {
            engine = std::make_unique<InvertedIndexEngine>();
        } else if (algorithmChoice == "trigram") {
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
            std::cout << "Document ID: " << doc->getId() << std::endl;
        }
    }
}
