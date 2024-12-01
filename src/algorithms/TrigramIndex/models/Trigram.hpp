//---------------------------------------------------------------------------
#ifndef TRIGRAM_HPP
#define TRIGRAM_HPP
//---------------------------------------------------------------------------
#include <cstdint>
#include <functional>
//---------------------------------------------------------------------------
struct Trigram {
    public:
    /// Constructor from raw value.
    explicit Trigram(uint32_t raw_value): value(raw_value) {}
    /// Constructor from char pointer.
    Trigram(const char* trigram, uint8_t word_offset)
        : value(
            (static_cast<uint32_t>(*trigram) << 24) | 
            (static_cast<uint32_t>(*(trigram+1)) << 16) |
            (static_cast<uint32_t>(*(trigram+2)) << 8) |
            word_offset
        ) {}

    /// Equality operator.
    bool operator==(const Trigram& other) const = default;
    /// Stream insertion operator.
    friend std::ostream& operator<<(std::ostream& os, const Trigram& trigram) {
        os << static_cast<char>((trigram.getRawValue() >> 24) & 0xFF)
            << static_cast<char>((trigram.getRawValue() >> 16) & 0xFF)
            << static_cast<char>((trigram.getRawValue() >> 8) & 0xFF);
        return os;
    }
    /// Get the trigram.
    [[nodiscard]] uint32_t get() const {  return (value >> 8); }
    /// Get the position within the word where the trigram starts.
    [[nodiscard]] uint8_t getWordOffset() const { return value & 0xFF; }
    /// Get the raw value.
    [[nodiscard]] uint32_t getRawValue() const { return value; }

    private:
    /// The raw value.
    uint32_t value;
};
//---------------------------------------------------------------------------
namespace std {
    template <>
    struct hash<Trigram> {
        std::size_t operator()(const Trigram& trigram) const noexcept {
            return std::hash<uint32_t>{}(trigram.get());
        }
    };
}
//---------------------------------------------------------------------------
#endif // TRIGRAM_HPP
//---------------------------------------------------------------------------