#ifndef PARALLEL_HASH_MAP_HPP
#define PARALLEL_HASH_MAP_HPP
//---------------------------------------------------------------------------
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
//---------------------------------------------------------------------------
#include "../../utils.hpp"
//---------------------------------------------------------------------------
template <typename Key, typename Value, size_t TableSize>
class ParallelHashTable {
 public:
  //---------------------------------------------------------------------------
  using Chain = std::vector<std::pair<Key, Value>>;
  using Bucket = std::pair<Chain, utils::SpinLock>;
  using Table = std::array<Bucket, utils::nextPowerOf2(TableSize)>;
  //---------------------------------------------------------------------------
  class TableIterator {
   public:
    using OuterIterator = typename Table::iterator;
    using InnerIterator = typename Chain::iterator;
    /// Constructor.
    explicit TableIterator(OuterIterator outer, OuterIterator outer_end, InnerIterator inner)
        : cur_outer(outer), outer_end(outer_end), cur_inner(inner) {
      advance_outer_if_needed();
    }
    /// Dereference Operator.
    std::pair<Key, Value>& operator*() const { return *cur_inner; }
    /// Member Access Operator.
    const std::pair<Key, Value>* operator->() const { return &*cur_inner; }
    /// Pre-Increment Operator.
    TableIterator& operator++() {
      ++cur_inner;
      advance_outer_if_needed();
      return *this;
    }
    /// Equality Operator.
    bool operator==(const TableIterator& other) const {
      return cur_outer == other.cur_outer && cur_inner == other.cur_inner;
    }
    /// Inequality Operator.
    bool operator!=(const TableIterator& other) const { return !(*this == other); }

   private:
    void advance_outer_if_needed() {
      while (cur_outer != outer_end && cur_inner == cur_outer->first.end()) {
        ++cur_outer;
        if (cur_outer != outer_end) {
          cur_inner = cur_outer->first.begin();
        }
      }
    }
    OuterIterator cur_outer;
    OuterIterator outer_end;
    InnerIterator cur_inner;
  };
  //---------------------------------------------------------------------------
  /// Default Constructor.
  ParallelHashTable() : table_mask(table.size() - 1) {}
  /// Move Constructor.
  ParallelHashTable(ParallelHashTable&& other) noexcept : table(std::move(other.table)) {}
  /// Move assignment.
  ParallelHashTable& operator=(ParallelHashTable&& other) noexcept {
    if (this != &other) {
      table = std::move(other.table);
    }
    return *this;
  }
  /**
   * Inserts the key value pair.
   * Threadsafe with concurrent inserts/updates
   * Not Threadsafe with concurrent reads
   * @param key Key
   * @param value Value
   */
  void insert(const Key key, const Value value) {
    auto& cur = table[hash(key)];
    std::unique_lock lck(cur.second);
    cur.first.push_back({key, value});
  }
  /**
   * Threadsafe with concurrent reads
   * Not Threadsafe with concurrent inserts/updates
   * @param key Key of the Key-Value-Pair
   * @return Corresponding value
   */
  Value* get(const Key& key) {
    auto& cur = table[hash(key)];
    for (auto& pair : cur.first) {
      if (pair.first == key) {
        return &pair.second;
      }
    }
    return nullptr;
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
    auto& cur = table[hash(key)];
    std::unique_lock lck(cur.second);

    for (auto& pair : cur.first) {
      if (pair.first == key) {
        update(pair.second);
        return;
      }
    }

    update(default_value);
    cur.first.push_back({key, default_value});
  }
  /**
   *
   * @param key Key
   * @return Iterator pointing to the Key,Value pair if it does not exist the end iterator is
   * returned
   */
  TableIterator find(const Key& key) {
    auto cur = table.begin() + hash(key);
    for (auto it = cur->first.begin(); it != cur->first.end(); ++it) {
      if (it->first == key) {
        return Iterator(cur, table.end(), it);
      }
    }
    return end();
  }
  /// The hash function for provided key on the table.
  size_t hash(const Key& k) const { return utils::mm_hash(k) & table_mask; }
  /// The begin-iterator for the hash table.
  TableIterator begin() {
    return TableIterator(table.begin(), table.end(), table.begin()->first.begin());
  }
  /// The end-iterator for the hash table.
  TableIterator end() {
    return TableIterator(table.end(), table.end(), table.begin()->first.begin());
  }
  /// The size of the hash table.
  uint32_t size() { return table.size(); }

 private:
  Table table;
  uint64_t table_mask;
};
//---------------------------------------------------------------------------
#endif  // PARALLEL_HASH_MAP_HPP