//---------------------------------------------------------------------------
#ifndef TRIGRAMUTILS_HPP
#define TRIGRAMUTILS_HPP
//---------------------------------------------------------------------------
#include <array>
//---------------------------------------------------------------------------
namespace utils {
//---------------------------------------------------------------------------
constexpr std::array<bool, 128> generateWhitelist() {
    std::array<bool, 128> whitelist = {false};
    for (char c = '0'; c <= '9'; ++c) {
        whitelist[static_cast<unsigned char>(c)] = true;
    }
    for (char c = 'A'; c <= 'Z'; ++c) {
        whitelist[static_cast<unsigned char>(c)] = true;
    }
    for (char c = 'a'; c <= 'z'; ++c) {
        whitelist[static_cast<unsigned char>(c)] = true;
    }
    whitelist[static_cast<unsigned char>('$')] = true;
    whitelist[static_cast<unsigned char>('%')] = true;
    whitelist[static_cast<unsigned char>('&')] = true;
    whitelist[static_cast<unsigned char>('+')] = true;
    whitelist[static_cast<unsigned char>('@')] = true;
    return whitelist;
}
//---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
#endif // TRIGRAMUTILS_HPP
//---------------------------------------------------------------------------