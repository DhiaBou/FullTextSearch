#ifndef FTS_UTILS_HPP
#define FTS_UTILS_HPP
//---------------------------------------------------------------------------
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
//---------------------------------------------------------------------------
#include <atomic>
#include <cstdint>
#include <string>
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
 * Implementation from:
 * https://en.cppreference.com/w/cpp/atomic/atomic_flag
 */
class SpinLock {
 public:
  void lock() noexcept {
    while (flag.test_and_set(std::memory_order_acquire)) flag.wait(true, std::memory_order_relaxed);
  }
  bool try_lock() noexcept { return !flag.test_and_set(std::memory_order_acquire); }
  void unlock() noexcept {
    flag.clear(std::memory_order_release);
    flag.notify_one();
  }

 private:
  std::atomic_flag flag{};
};
//---------------------------------------------------------------------------
/**
 *  Bithack from:
 *  https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
 */
static constexpr uint64_t nextPowerOf2(uint64_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  v++;
  return v;
}
//---------------------------------------------------------------------------
/**
 *  MurmurHash64A
 */
static constexpr uint64_t mm_hash(uint64_t k) {
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;
  uint64_t h = 0x8445d61a4e774912 ^ (8 * m);
  k *= m;
  k ^= k >> r;
  k *= m;
  h ^= k;
  h *= m;
  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;
}
//---------------------------------------------------------------------------
}  // namespace utils
//---------------------------------------------------------------------------
#endif  // FTS_UTILS_HPP