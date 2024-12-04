#ifndef FTS_ENGINE_HPP
#define FTS_ENGINE_HPP

#include <string>
#include <vector>
#include <cstdint>
#include "documents/document.hpp"
#include "documents/document_iterator.hpp"

/**
 * @brief Interface for full-text search engines.
 *
 * This abstract class defines the interface for full-text search engines.
 * Implementations of this interface should provide methods to build an index
 * from a collection of documents and to search for documents matching a given query.
 */
class FullTextSearchEngine {
public:
    /**
     * Destructor.
     */
    virtual ~FullTextSearchEngine() = default;
    /**
     * @brief Builds an index from the given documents.
     *
     * This method processes the documents provided by the DocumentIterator
     * and builds an index to support efficient full-text search.
     *
     * @param it DocumentIterator providing access to the documents to be indexed.
     */
    virtual void indexDocuments(DocumentIterator it) = 0;
    /**
     * @brief Searches for documents matching the given query.
     *
     * This method searches the indexed documents for matches to the given query
     * and returns a list of matching document IDs.
     *
     * @param query The search query as a string.
     * @return A vector of document IDs that match the query.
     */
    virtual std::vector<uint32_t> search(const std::string &query) = 0;
};

#endif  // FTS_ENGINE_HPP
