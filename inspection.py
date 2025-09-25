import sys
import math
import csv

if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    with open(in_file, 'r') as f_in:
        reader = csv.reader(f_in, delimiter = "\t")
        next(reader)
        train_labels = []
        for row in reader:
            train_labels.append(row[-1])

        train_zeros = train_labels.count('0')
        train_ones = train_labels.count('1')
        train_total = train_ones + train_zeros

        p0 = train_zeros / train_total
        p1 = train_ones / train_total

        entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        error = min(train_ones, train_zeros) / train_total

    with open(out_file, 'w') as f_out:
        f_out.write(f'entropy: {entropy:.6f}\n')
        f_out.write(f'error: {error:.6f}\n')