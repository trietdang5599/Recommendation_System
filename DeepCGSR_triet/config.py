
class config:
    dataset_name='movielens1M'
    dataset_path='data/ratings.dat'
    # dataset_name='movielens20M'
    # dataset_path='data/ratings_20M.dat'
    # dataset_name='reviewAmazon'
    # dataset_path='data/ratings_AB.csv'
    model_name='deepcgsr'
    epoch=100
    learning_rate=0.01
    batch_size=200
    weight_decay=1e-6
    device='cuda:0'
    save_dir='chkpt'

args = config()