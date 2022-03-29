from engines.data import DataManager
from engines.utils.logger import get_logger
from engines.train import train
from engines.predict import Predictor
from engines.utils.word2vec import Word2VecUtils
from engines.utils.Glove import GloveUtils
from engines.utils.FastText import FastTextUtils
from config import mode, classifier_config, word2vec_config, Glove_config, FastText_config, CUDA_VISIBLE_DEVICES
import json
import os


if __name__ == '__main__':
    logger = get_logger('./logs')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
    if mode == 'train_classifier':
        logger.info(json.dumps(classifier_config, indent=2))
        data_manage = DataManager(logger)
        logger.info('mode: train_classifier')
        logger.info('model: {}'.format(classifier_config['classifier']))
        train(data_manage, logger)
    elif mode == 'interactive_predict':
        logger.info(json.dumps(classifier_config, indent=2))
        data_manage = DataManager(logger)
        logger.info('mode: predict_one')
        logger.info('model: {}'.format(classifier_config['classifier']))
        predictor = Predictor(data_manage, logger)
        predictor.predict_one('warm start')
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            results = predictor.predict_one(sentence)
            print(results)
    elif mode == 'train_word2vec':
        logger.info(json.dumps(word2vec_config, indent=2))
        logger.info('mode: train_word2vec')
        w2v = Word2VecUtils(logger)
        w2v.train_word2vec()
    elif mode == 'train_Glove':
        logger.info(json.dumps(Glove_config, indent=2))
        logger.info('mode: train_Glove')
        glove = GloveUtils(logger)
        glove.TrainGlove()
    elif mode == 'train_FastText':
        logger.info(json.dumps(FastText_config, indent=2))
        logger.info('mode: train_FastText')
        ft = FastTextUtils(logger)
        ft.TrainFastText()
    elif mode == 'test':
        logger.info('mode: test')
        data_manage = DataManager(logger)
        predictor = Predictor(data_manage, logger)
        predictor.predict_test()
    elif mode == 'save_model':
        logger.info('mode: save_pb_model')
        data_manage = DataManager(logger)
        predictor = Predictor(data_manage, logger)
        predictor.save_model()
