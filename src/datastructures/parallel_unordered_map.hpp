#ifndef PARALLEL_UNORDERED_MAP_HPP
#define PARALLEL_UNORDERED_MAP_HPP
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

template <typename Key, typename Value, size_t buckets>
class ParallelUnorderedMap {
  typedef std::vector<std::pair<Key, Value>> Chain;
  typedef std::pair<Chain, std::mutex> Bucket;
  typedef std::array<std::pair<Chain, std::mutex>, buckets> Buckets;

  std::array<std::pair<Chain, std::mutex>, buckets> elements;
  std::atomic<int> num_elements;

  size_t hash(const Key& k) const { return std::hash<Key>()(k) % elements.size(); }

 public:
  class Iterator {
    using OuterIterator = typename Buckets::iterator;
    using InnerIterator = typename Chain::iterator;

    OuterIterator cur_outer;
    OuterIterator outer_end;
    InnerIterator cur_inner;

    void advance_outer_if_needed() {
      while (cur_outer != outer_end && cur_inner == cur_outer->first.end()) {
        ++cur_outer;
        if (cur_outer != outer_end) {
          cur_inner = cur_outer->first.begin();
        }
      }
    }

   public:
    explicit Iterator(OuterIterator outer, OuterIterator outer_end, InnerIterator inner)
        : cur_outer(outer), outer_end(outer_end), cur_inner(inner) {
      advance_outer_if_needed();
    }

    std::pair<Key, Value>& operator*() const { return *cur_inner; }

    const std::pair<Key, Value>* operator->() const { return &*cur_inner; }

    Iterator& operator++() {
      ++cur_inner;
      advance_outer_if_needed();
      return *this;
    }

    bool operator==(const Iterator& other) const {
      return cur_outer == other.cur_outer && cur_inner == other.cur_inner;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }
  };

  /**
   * Inserts the key value pair.
   * Threadsafe with concurrent inserts/updates
   * Not Threadsafe with concurrent reads
   * @param key Key
   * @param value Value
   */
  void insert(const Key key, const Value value) {
    auto& cur = elements[hash(key)];

    std::unique_lock lck(cur.second);
    cur.first.push_back({key, value});

    ++num_elements;
  }

  /**
   * Threadsafe with concurrent reads
   * Not Threadsafe with concurrent inserts/updates
   * @param key Key of the Key-Value-Pair
   * @return Corresponding value
   */
  std::optional<Value> get(const Key& key) const {
    const auto& cur = elements[hash(key)];
    for (const auto& pair : cur.first) {
      if (pair.first == key) {
        return pair.second;
      }
    }

    return std::nullopt;
  }

  /**
   * Updates the value of the corresponding key using the update Functor.
   * If the key is not in the map a new key value pair (Key, update(default_value)) is inserted
   * Threadsafe with concurrent inserts/updates
   * Not Threadsafe with concurrent reads
   * @tparam Functor
   * @param key
   * @param update
   * @param default_value
   */
  template <typename Functor>
  void updateOrInsert(const Key& key, Functor update, Value default_value) {
    auto& cur = elements[hash(key)];

    std::unique_lock lck(cur.second);

    for (auto& pair : cur.first) {
      if (pair.first == key) {
        update(pair.second);
        return;
      }
    }

    update(default_value);
    cur.first.push_back({key, default_value});
    ++num_elements;
  }

  /**
   *
   * @return Iterator pointing to the first entry
   */
  Iterator begin() {
    return Iterator(elements.begin(), elements.end(), elements.begin()->first.begin());
  }

  /**
   *
   * @return Iterator pointing to the element one after the last entry
   */
  Iterator end() {
    return Iterator(elements.end(), elements.end(), elements.begin()->first.begin());
  }

  /**
   *
   * @param key Key
   * @return Iterator pointing to the Key,Value pair if it does not exist the end iterator is
   * returned
   */
  Iterator find(const Key& key) {
    auto cur = elements.begin() + hash(key);
    for (auto it = cur->first.begin(); it != cur->first.end(); ++it) {
      if (it->first == key) {
        return Iterator(cur, elements.end(), it);
      }
    }
    return end();
  }

  /**
   *
   * @return no entry in the parallel_unordered_map
   */
  [[nodiscard]] bool empty() const { return num_elements == 0; }

  /**
   *
   * @return Number of key,value pairs
   */
  [[nodiscard]] size_t size() const { return num_elements; }
};

#endif  // PARALLEL_UNORDERED_MAP_HPP
