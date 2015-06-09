import numpy

from nmt import train

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    trainerr, validerr, testerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        encoder='gru',
                                        decoder='gru_cond_simple',
                                        hiero=None, #'gru_hiero', # or None
                                        n_words_src=params['n-words-src'][0],
                                        n_words=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        alpha_c=params['alpha-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=100,
                                        batch_size=64,
                                        valid_batch_size=64,
                                        validFreq=1000,
                                        dispFreq=1,
                                        saveFreq=500,
                                        sampleFreq=10,
                                        dataset='stan',
                                        dictionary='./stan/vocab_and_data_sub_europarl/vocab_sub_europarl.fr.pkl',
                                        dictionary_src='./stan/vocab_and_data_sub_europarl/vocab_sub_europarl.en.pkl',
                                        use_dropout=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_gru_sub_europarl_enfr.npz'],
        'dim_word': [50],
        'dim': [50],
        'n-words': [30000], 
        'n-words-src': [30000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'alpha-c': [0.], 
        'use-dropout': [0],
        'learning-rate': [0.0001],
        'reload': [False]})


