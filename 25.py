from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd
import argparse


def status(message):
    print(message)


def read_file(path):
    status("Reading file ...")
    try:
        reader = pd.read_csv(path, header=None)
        return reader
    except:
        print("File not found!")


def write_file(path, file, data):
    status("Writing " + file + " file ...")
    try:
        data.to_csv(path, index=False, header=None)
    except:
        print("Cannot write a file!")


def convert_dataset(dataset):
    status('Converting dataset to transaction...')
    data = dataset.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


def frequent_itemsets(dataset, minSupport):
    frequent_itemsets = apriori(dataset, min_support=minSupport, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    return frequent_itemsets


def write_fi_file(itemsets, path):
    summary = itemsets.groupby(by='length').count()
    data = []
    for index, row in summary.iterrows():
        data.append(row['support'])
        items = itemsets[(itemsets['length'] == index)]
        for i, r in items.iterrows():
            strs = str(round(r['support'], 2)) + ' '
            for item in r['itemsets']:
                strs = strs + item + ' '

            data.append(strs)
    df = pd.DataFrame(data, columns=['A'])
    write_file(path, "FI", df)


def main(args):
    data = read_file(args.input)
    if data is not None:
        minSupport = float(args.minsup)
        minConf = float(args.minconf)
        data = convert_dataset(data)
        data = frequent_itemsets(data, minSupport)
        write_fi_file(data, args.outFI)
        status("Finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Association rules')
    parser.add_argument('input', help='input file path')
    parser.add_argument('outFI', help='output FI file path')
    parser.add_argument('outAR', help='output AR file path')
    parser.add_argument('minsup', help='min support')
    parser.add_argument('minconf', help='min confidence')
    args = parser.parse_args()
    main(args)
