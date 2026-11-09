GROCERIES_PATH = "../../data/groceries.csv"

def load_transactions(path=GROCERIES_PATH):
    transactions = []

    # the csv doesn't have a header and each row has a different number of products, so it's easier to read it line by line instead of using pandas
    with open(path) as file:
        for line in file:
            # we use split to separate the products by commas
            # we use strip to remove the spacing and extra spaces (some products have extra spaces at the end)
            products = [product.strip() for product in line.split(',')]
            # set removes repeated products in the same transaction, sorted so each pair is in the same order
            transactions.append(sorted(set(products)))

    return transactions

def count_pairs(transactions):
    pair_counts = {}

    for transaction in transactions:
        # every combination of two products in the transaction
        for i in range(len(transaction)):
            for j in range(i + 1, len(transaction)):
                pair = (transaction[i], transaction[j])
                # if the pair isn't in the dict yet it starts at 0
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts


if __name__ == "__main__":
    transactions = load_transactions()
    pair_counts = count_pairs(transactions)

    # 10 most frequent pairs
    top_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
    for pair, count in top_pairs[:10]:
        print(pair, count)
