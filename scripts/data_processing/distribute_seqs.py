import sys


def labels_to_dict(labels_file):
    labels_dict = {}
    with open(labels_file, "r") as fin:
        for line in fin:
            coords = line.strip().split(' ')[-1]
            chrom, start, end = coords.split('-')
            key = f'{chrom}:{start}-{end}'
            labels_dict[key] = line.strip()
    return labels_dict


def main():

    labels_dict = labels_to_dict(sys.argv[1])
    train_split = open('train_split.csv', 'w')
    val_spilt = open('val_split.csv', 'w')
    test_split = open('test_split.csv', 'w')

    with open(sys.argv[2], 'r') as fin:
        for line in fin:
            coord, split_type = line.strip().split('\t')

            if coord in labels_dict:

                if split_type == 'train':
                    train_split.write(f'{labels_dict[coord]}\n')
                elif split_type == 'val':
                    val_spilt.write(f'{labels_dict[coord]}\n')
                elif split_type == 'test':
                    test_split.write(f'{labels_dict[coord]}\n')
                else:
                    raise ValueError('Invalid split type')

            else:
                print(f'Warning: {coord} not found in labels_dict')

    train_split.close()
    val_spilt.close()
    test_split.close()


if __name__ == '__main__':
    main()
