import numpy

from nmt import train

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    trainerr, validerr, testerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        n_words_src=params['n-words-src'][0],
                                        decay_c=params['decay-c'][0],
                                        alpha_c=params['alpha-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        encoder='gru',
                                        decoder='gru_cond', #'gru_cond_simple',
                                        maxlen=30,
                                        batch_size=128,
                                        valid_batch_size=128,
                                        validFreq=1000,
                                        dispFreq=1,
                                        saveFreq=500,
                                        sampleFreq=500,
                                        dataset='trans_enhi', 
                                        dictionary='/data/lisatmp3/chokyun/transliteration/TranslitDataset/vocab.hi.pkl',
                                        dictionary_src='/data/lisatmp3/chokyun/transliteration/TranslitDataset/vocab.en.pkl',
                                        use_dropout=True if params['use-dropout'][0] else False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_lstm_enhi.npz'],
        'dim_word': [32],
        'dim': [512],
        'n-words': [82], 
        'n-words-src': [41], 
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'alpha-c': [0.], 
        'use-dropout': [0],
        'learning-rate': [0.0001],
        'reload': [False]})

