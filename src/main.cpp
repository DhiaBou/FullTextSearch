#include <iostream>

#include "FullTextSearchEngine.hpp"
#include "algorithms/DummySearch/DummySearchEngine.hpp"
#include "algorithms/InvertedIndex/InvertedIndexEngine.hpp"
#include "algorithms/TrigramIndex/TrigramIndexEngine.hpp"
#include "algorithms/VectorSpaceModel/VectorSpaceModelEngine.hpp"
#include "dataUtils/Document.hpp"
#include "dataUtils/DocumentIterator.hpp"
#include "dataUtils/DocumentUtils.hpp"
#include <string>

int main() {
    // Choose search engine algorithm
    FullTextSearchEngine *engine = nullptr;
    std::string algorithmChoice;


    std::string directoryPath;
    std::cout << "Enter search directory absolute path: ";
    std::cin >> directoryPath;

    DocumentIterator it(directoryPath);

    do {
        std::cout << "Select search algorithm (vsm/inverted/trigram/dummy): ";
        std::cin >> algorithmChoice;
        if (algorithmChoice == "vsm") {
            engine = new VectorSpaceModelEngine();
        } else if (algorithmChoice == "inverted") {
            engine = new InvertedIndexEngine();
        } else if (algorithmChoice == "trigram") {
            engine = new TrigramIndexEngine();
        } else if (algorithmChoice == "dummy") {
            engine = new DummySearchEngine();
        } else {
            std::cout << "Invalid choice!" << std::endl;
        }
    } while (engine == nullptr);

     engine->indexDocuments(std::move(it));

     std::string query;
     while (true) {
         std::cout << "Enter search query (or 'exit' to quit): ";
         std::cin >> query;
         if (query == "exit") break;

         auto results = engine->search(query);

         for (const auto &doc: results) {
             std::cout << "Document ID: " << doc->getId() << std::endl;
         }
     }

    delete engine;
    return 0;
}
