#ifndef TRIGRAM_HPP
#define TRIGRAM_HPP
//---------------------------------------------------------------------------
#include <cstdint>
#include <functional>
#include <ostream>
//---------------------------------------------------------------------------
namespace trigramlib {
//---------------------------------------------------------------------------
constexpr uint8_t kMaxWordOffset = 15;
//---------------------------------------------------------------------------
struct Trigram {
 public:
  /// Default constructor.
  Trigram() : value(0) {}
  /// Constructor from raw value.
  explicit Trigram(uint32_t raw_value) : value(raw_value) {}
  /// Constructor from char pointer.
  Trigram(const char* trigram, uint8_t word_offset)
      : value((static_cast<uint32_t>(*trigram) << 24) |
              (static_cast<uint32_t>(*(trigram + 1)) << 16) |
              (static_cast<uint32_t>(*(trigram + 2)) << 8) | word_offset) {}

  /// Equality operator.
  /// Note: Two trigrams are equal if they consist of the same 3 letters.
  /// Thus, the word offset is not considered for the equality.
  bool operator==(const Trigram& other) const { return get() == other.get(); };
  /// Stream insertion operator.
  friend std::ostream& operator<<(std::ostream& os, const Trigram& trigram) {
    os << static_cast<char>((trigram.getRawValue() >> 24) & 0xFF)
       << static_cast<char>((trigram.getRawValue() >> 16) & 0xFF)
       << static_cast<char>((trigram.getRawValue() >> 8) & 0xFF);
    return os;
  }
  /// Get the trigram.
  [[nodiscard]] uint32_t get() const { return (value >> 8); }
  /// Get the position within the word where the trigram starts.
  [[nodiscard]] uint8_t getWordOffset() const { return value & 0xFF; }
  /// Set the position within the word where the trigram starts.
  void setWordOffset(const uint8_t offset) { value = (value & 0xFFFFFF00) | offset; }
  /// Get the raw value.
  [[nodiscard]] uint32_t getRawValue() const { return value; }

 private:
  /// The raw value.
  uint32_t value;
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
namespace std {
//---------------------------------------------------------------------------
template <>
struct hash<trigramlib::Trigram> {
  std::size_t operator()(const trigramlib::Trigram& trigram) const noexcept {
    return std::hash<uint32_t>{}(trigram.get());
  }
};
//---------------------------------------------------------------------------
}  // namespace std
//---------------------------------------------------------------------------
#endif  // TRIGRAM_HPP