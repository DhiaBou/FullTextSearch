// ---------------------------------------------------------------------------
#ifndef TRIGRAMUTILS_HPP
#define TRIGRAMUTILS_HPP
//---------------------------------------------------------------------------
#include <cstdint>
#include <functional>
//---------------------------------------------------------------------------
struct Trigram {
    public:
    /// Constructor.
    explicit Trigram(uint32_t raw_value): value(raw_value) {}
    /// Constructor.
    Trigram(uint32_t trigram, uint8_t wordOffset)
        : value((trigram << 8) | (wordOffset & 0xFF)) {
    }

    /// Get the trigram
    [[nodiscard]] uint64_t get() const {  return (value >> 8); }
    /// Get the position within the word where the trigram starts
    [[nodiscard]] uint16_t getWordOffset() const { return value & 0xFF; }
    /// Get the raw value
    [[nodiscard]] uint32_t getValue() const { return value; }

    private:
    /// The raw value
    uint32_t value;
};
//---------------------------------------------------------------------------
#endif // TRIGRAMUTILS_HPP
//---------------------------------------------------------------------------