#ifndef FTS_ENGINE_HPP
#define FTS_ENGINE_HPP

#include <string>
#include <vector>
#include "documents/document.hpp"
#include "documents/document_iterator.hpp"

/**
 * Interface to be implemented by the full-text search engines.
 */
class FullTextSearchEngine {
public:
    /// Destructor.
    virtual ~FullTextSearchEngine() = default;
    /// Builds an index on the given documents.
    virtual void indexDocuments(DocumentIterator it) = 0;
    /// Searches matching documents for the given text.
    virtual std::vector<std::shared_ptr<Document>> search(const std::string &query) = 0;
};

#endif  // FTS_ENGINE_HPP
