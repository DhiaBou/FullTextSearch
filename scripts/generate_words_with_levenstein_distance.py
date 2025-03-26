import sys

def levenshtein_variants(word, max_distance):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    current_words = {word}

    for _ in range(max_distance):
        next_words = set()

        for word in current_words:
            word_length = len(word)

            # Generate words by deleting one character
            for i in range(word_length):
                next_words.add(word[:i] + word[i+1:])

            # Generate words by replacing one character
            for i in range(word_length):
                for char in alphabet:
                    if char != word[i]:  # Avoid replacing with the same letter
                        next_words.add(word[:i] + char + word[i+1:])

            # Generate words by inserting one character
            for i in range(word_length + 1):
                for char in alphabet:
                    next_words.add(word[:i] + char + word[i:])

        # Move to next level
        current_words = next_words

    return current_words


# Ensure correct command-line arguments
if len(sys.argv) != 4:
    print("Usage: python levenshtein.py <word> <distance> <output_file>")
    sys.exit(1)

# Get inputs from command line
target_word = sys.argv[1]
output_file = sys.argv[3]

try:
    max_distance = int(sys.argv[2])
    if max_distance < 1:
        raise ValueError
except ValueError:
    print("Error: Distance must be a positive integer.")
    sys.exit(1)

# Find words within the given Levenshtein distance
words_within_distance = levenshtein_variants(target_word, max_distance)

# Write all words to a single file, separated by line breaks
with open(output_file, "w") as file:
    file.write("\n".join(words_within_distance))
