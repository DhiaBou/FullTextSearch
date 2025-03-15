#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
//---------------------------------------------------------------------------
#include <charconv>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

//---------------------------------------------------------------------------
namespace utils {
//---------------------------------------------------------------------------
/**
 * @brief A utility for reading memory mapped files.
 */
class FileReader {
 public:
  explicit FileReader(const char *path) {
    fd = open(path, O_RDONLY);
    if (fd < 0) exit(1);
    size = lseek(fd, 0, SEEK_END);
    data = static_cast<char *>(mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0));
    if (data == MAP_FAILED) {
      close(fd);
      exit(1);
    }
  }
  ~FileReader() {
    munmap(data, size);
    close(fd);
  }
  FileReader(const FileReader &other) = delete;
  FileReader &operator=(const FileReader &other) = delete;

  const char *begin() const { return data; }
  const char *end() const { return data + size; }
  uint64_t getSize() const { return size; }

 private:
  int fd;
  uint64_t size;
  char *data;
};
//---------------------------------------------------------------------------
/**
 * @brief A utility to merge multiple unordered maps into a single one.
 * To be precise, merges the given maps into the one located at index 0.
 * @param maps The maps to be merged.
 */
template <typename K>
void merge(std::vector<std::unordered_map<K, uint32_t>> &maps) {
  if (maps.empty()) return;
  auto &main_index = maps[0];
  for (size_t i = 1; i < maps.size(); ++i) {
    for (auto &[key, value] : maps[i]) {
      main_index[key] += value;
    }
  }
}
//---------------------------------------------------------------------------
}  // namespace utils