#ifndef FTS_ENGINE_HPP
#define FTS_ENGINE_HPP

#include <string>
#include <vector>
#include "documents/document.hpp"
#include "documents/document_iterator.hpp"

class FullTextSearchEngine {
public:
    virtual void indexDocuments(DocumentIterator it) = 0;

    virtual std::vector<std::shared_ptr<Document>> search(const std::string &query) = 0;

    virtual ~FullTextSearchEngine() = default;
};

#endif  // FTS_ENGINE_HPP
