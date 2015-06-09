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
                                        n_words_src=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=20,
                                        batch_size=16,
                                        valid_batch_size=16,
                                        validFreq=1000,
                                        dispFreq=1,
                                        saveFreq=1000,
                                        sampleFreq=1000,
                                        dataset='wmt14enfr', 
                                        dictionary='/data/lisatmp3/chokyun/wmt14/parallel-corpus/en-fr/vocab.fr.pkl',
                                        use_dropout=True if params['use-dropout'][0] else False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_lstm.npz'],
        'dim_word': [64],
        'dim': [128],
        'n-words': [20000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'use-dropout': [0],
        'learning-rate': [0.0001],
        'reload': [False]})

