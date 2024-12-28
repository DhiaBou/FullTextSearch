import csv
import sys

def read_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    return data

def count_matches(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    cols_file1 = sorted(data1, key=lambda x: x[0])
    cols_file2 = sorted(data2, key=lambda x: x[0])

    matches = 0
    sum_abs_diff = 0
    i, j = 0, 0
    while i < len(cols_file1) and j < len(cols_file2):
        if cols_file1[i][0] == cols_file2[j][0]:
            matches += 1
            sum_abs_diff += abs(float(cols_file1[i][1]) - float(cols_file2[j][1]))
            i += 1
            j += 1
        elif cols_file1[i][0] < cols_file2[j][0]:
            i += 1
        else:
            j += 1

    return matches, sum_abs_diff

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_csv.py <file1.csv> <file2.csv>")
        print("Expects two csv files without header in the following format: <docId>,<score>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    matches, sum_abs_diff = count_matches(file1, file2)
    print(f"Number of document appearing in both searches: {matches}")
    print(f"Sum of absolute differences in the second column for matching entries: {sum_abs_diff}")