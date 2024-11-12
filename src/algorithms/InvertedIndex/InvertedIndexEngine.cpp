// Created by fts on 10/31/24.

#include "InvertedIndexEngine.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

void InvertedIndexEngine::indexDocuments(DocumentIterator doc_it) {
    do {
        auto doc = *doc_it;
        auto begin = doc->getBegin();
        auto end = begin + doc->getSize();

        auto it = begin;

        while (it < end) {
            while (((*it >= 'a' && *it <= 'z') || (*it >= 'A' && *it <= 'Z')) && it < end) {
                ++it;
            }
            std::string word(begin, it);
            std::transform(word.begin(), word.end(), word.begin(), tolower);
            index[std::string(begin, it)][doc->getId()]++;
            it++;
            begin = it;
        }

        ++doc_it;
    } while (doc_it.hasNext());
}

std::vector<std::shared_ptr<Document>> InvertedIndexEngine::search(const std::string &query) {
    std::string lower_case_query;
    std::transform(query.begin(), query.end(), lower_case_query.begin(), tolower);

    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, std::greater<>> results;

    for (auto potential_result: index[lower_case_query]) {
        if (results.size() < 10) {
            results.emplace(potential_result.second, potential_result.first);
        } else if (potential_result.first > results.top().first) {
            results.pop();
            results.emplace(potential_result.second, potential_result.first);
        }
    }

    size_t place = results.size();
    while (!results.empty()) {
        auto result = results.top();
        results.pop();
        std::cout << "Document " << result.second << " has occured " << result.first << " times" << std::endl;
    }

    return {};
}
