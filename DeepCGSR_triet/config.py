
class config:
    # dataset_name='movielens1M'
    # dataset_path='data/ratings.dat'
    # dataset_name='movielens20M'
    # dataset_path='data/ratings_20M.dat'
    dataset_name='reviewAmazon'
    dataset_json = 'data/DigitalMusic/Digital_Music_Filtered.json'
    # dataset_json = 'data/DigitalMusic/Digital_Music_Raw.json'
    # dataset_json = 'data/raw/All_Beauty_Filtered.json'
    # dataset_json = 'data/DigitalMusic/Digital_Music_Filtered_7.json'
    user_length = 0
    item_length = 0
    # data_path='feature/filteredFeatures.csv'
    data_path='feature/allFeatureReview.csv'
    data_feature='data/final_data_feature.csv'
    model_name='deepcgsr'
    epoch=100
    learning_rate=0.01
    batch_size=32
    weight_decay=1e-6
    device='cuda:0'
    save_dir='chkpt'
    isRemoveOutliner = False

args = config()