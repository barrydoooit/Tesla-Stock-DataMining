from preprocessing import read_data, parse_data
from Apriori import intra_stock
if __name__ == '__main__':
    dfall = read_data()

    dfall = parse_data(dfall)
    print(dfall.info())
    intra_stock(dfall.symbol_1.to_numpy(), 1)
