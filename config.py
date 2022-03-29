# [train_word2vec,train_Glove,train_FastText,train_classifier, interactive_predict, save_model, test]
mode = 'train_classifier'

word2vec_config = {
    'stop_words': 'data/w2v_data/stop_words.txt', 
    'train_data': 'data/w2v_data/dataset.csv',
    'model_dir': 'model/word2vec_model',
    'model_name': 'word2vec_model.pkl',
    'word2vec_dim': 180,
    'min_count': 3,
    'epoch': 120,
    'workers': 10,
    'sg': 'skip-gram'
}

Glove_config = {
    'window': 5,
    'epoch': 120,
    'workers': 10,
    'learning_rate': 0.001,
    'train_data': 'data/g_data/dataset.csv',
    'model_dir': 'model/Glove_model/',
    'model_name': 'Glove_model.pkl',
    'corpus_model':'Corpus_model.pkl',
    'tokenizer_name':'tokenizer.pkl',
    'components': 180,
}
FastText_config = {
    'window': 5,
    'epoch': 120,
    'stop_words': 'data/f_data/stop_words.txt',
    'train_data': 'data/f_data/dataset.csv',
    'model_dir': 'model/FastText_model/',
    'model_name': 'FastText_model.pkl',
    'size': 180,
    'min_count': 3,
    'algorithm': 'skip-gram',
    'n_workers': 10,
    'seed': 1
}

CUDA_VISIBLE_DEVICES = 0

classifier_config = {
    'classifier': 'transformer',
    'bert_op': 'bert-base-cased',
    'train_file': 'data/train_dataset.csv',
    'train_file_folder': 'data',
    'train_file_name': 'train_dataset.csv',
    'val_file': 'data/val_dataset.csv',
    'test_file': 'data/test_dataset.csv',
    'embedding_method': 'word2vec',
    'token_level': 'word',
    'embedding_dim': 180,
    'token_file': 'data/word-token2id',
    'classes': {'vul': 0, 'nvul': 1},
    'checkpoints_dir': 'model/transformer-word',
    'checkpoint_name': 'transformer-word',
    'num_filters': 128,
    'learning_rate': 0.01,
    'epoch': 20,
    'max_to_keep': 1,
    'print_per_batch': 20,
    'is_early_stop': False,
    'use_attention': True,
    'attention_size': 180,
    'patient': 10,
    'batch_size': 64,
    'max_sequence_length': 500,
    'dropout_rate': 0.3,
    'hidden_dim': 2048,
    'encoder_num': 1,
    'head_num': 12,
    'metrics_average': 'binary',
    'use_focal_loss': False,
    'use_gan': True,
    'gan_method': 'fgm',
    'use_r_drop': False
}
