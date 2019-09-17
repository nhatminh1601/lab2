from efficient_apriori import apriori as eff_apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

dataset = pd.read_csv('weather.nominal.arff.csv', header=None)
transactions = []
te = TransactionEncoder()
data = dataset.values.tolist()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
df

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
test = frequent_itemsets[(frequent_itemsets['length'] == 1) &
                         (frequent_itemsets['support'] >= 0.1)]
print(frequent_itemsets['length'])
print(frequent_itemsets)
summary = frequent_itemsets.groupby(by='length').count()

data = []
for index, row in summary.iterrows():
    print(row['support'])
    items = frequent_itemsets[(frequent_itemsets['length'] == index)]
    for i, r in items.iterrows():
        stra = str(round(r['support'], 2)) + ' '
        for item in r['itemsets']:
            stra = stra + item + ' '

        print(stra)
a=association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
# print(a)
# print(test)


for i in range(0, 15):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 4)])


def exec_apriori(tr):
    itemsets, rules = eff_apriori(tr, min_support=0.1, min_confidence=0.9)
    return itemsets, rules


item_eff_apriori, rules_eff_apriori = exec_apriori(transactions)

# rules_eff_apriori = sorted(rules_eff_apriori, key=lambda x: x.confidence, reverse=True)
print(rules_eff_apriori)
