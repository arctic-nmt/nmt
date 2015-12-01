'''
Build a attention-based neural machine translation model
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from scipy import optimize, stats
from collections import OrderedDict
#from sklearn.cross_validation import KFold

import wmt14enfr
import iwslt14zhen
import openmt15zhen
import trans_enhi
import stan

profile = False

# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = {'wmt14enfr': (wmt14enfr.load_data, wmt14enfr.prepare_data),
            'iwslt14zhen': (iwslt14zhen.load_data, iwslt14zhen.prepare_data),
            'openmt15zhen': (openmt15zhen.load_data, openmt15zhen.prepare_data),
            'trans_enhi': (trans_enhi.load_data, trans_enhi.prepare_data),
            'stan': (stan.load_data, stan.prepare_data),
            }

def get_dataset(name):
    return datasets[name][0], datasets[name][1]

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
            state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive'%kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'ff_nb': ('param_init_fflayer_nb', 'fflayer_nb'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cond_simple': ('param_init_gru_cond_simple', 'gru_cond_simple_layer'),
          'gru_hiero': ('param_init_gru_hiero', 'gru_hiero_layer'),
          'rnn': ('param_init_rnn', 'rnn_layer'),
          'rnn_cond': ('param_init_rnn_cond', 'rnn_cond_layer'),
          'rnn_hiero': ('param_init_rnn_hiero', 'rnn_hiero_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def linear(x):
    return x

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# feedforward layer with no bias: affine transformation + point-wise nonlinearity
def param_init_fflayer_nb(options, params, prefix='ff_nb', nin=None, nout=None, ortho=True):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)

    return params

def fflayer_nb(tparams, state_below, options, prefix='ff_nb', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')]))

# RNN layer
def param_init_rnn(options, params, prefix='rnn', nin=None, dim=None):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def rnn_layer(tparams, state_below, options, prefix='rnn', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[0]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step(m_, xx_, h_, Ux):
        preactx = tensor.dot(h_, Ux)
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_belowx],
                                outputs_info = [tensor.alloc(0., n_samples, dim)],
                                                #None, None, None, None],
                                non_sequences=[tparams[_p(prefix, 'Ux')]], 
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)
    rval = [rval]
    return rval

# Conditional RNN layer with Attention
def param_init_rnn_cond(options, params, prefix='rnn_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']

    params = param_init_rnn(options, params, prefix, nin=nin, dim=dim)

    # context to LSTM
    Wcx = norm_weight(dimctx,dim)
    params[_p(prefix,'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin,dimctx)
    params[_p(prefix,'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params

def rnn_cond_layer(tparams, state_below, options, prefix='rnn', 
                   mask=None, context=None, one_step=False, 
                   init_memory=None, init_state=None, 
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[0]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix,'b_att')]
    pctx_ += tparams[_p(prefix,'b_att')]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_belowx += tparams[_p(prefix, 'bx')]
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, xx_, xc_, h_, ctx_, alpha_, pctx_,
              Wd_att, U_att, c_tt, Ux, Wcx):
        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pctx_ + pstate_[None,:,:] 
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:,:,None]).sum(0) # current context

        preactx = tensor.dot(h_, Ux)
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        h = tensor.tanh(preactx)

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, ctx_, alpha.T #, pstate_, preact, preactx, r, u

    if one_step:
        rval = _step(mask, state_belowx, state_belowc, init_state, None, None, 
                     pctx_, tparams[_p(prefix,'Wd_att')],
                     tparams[_p(prefix,'U_att')],
                     tparams[_p(prefix, 'c_tt')],
                     tparams[_p(prefix, 'Ux')],
                     tparams[_p(prefix, 'Wcx')] )
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=[mask, state_belowx, state_belowc],
                                    outputs_info = [init_state, 
                                                    tensor.alloc(0., n_samples, context.shape[2]),
                                                    tensor.alloc(0., n_samples, context.shape[0])],
                                                    #None, None, None, 
                                                    #None, None],
                                    non_sequences=[pctx_,
                                                   tparams[_p(prefix,'Wd_att')],
                                                   tparams[_p(prefix,'U_att')],
                                                   tparams[_p(prefix, 'c_tt')],
                                                   tparams[_p(prefix, 'Ux')],
                                                   tparams[_p(prefix, 'Wcx')]
                                                   ],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile)
    return rval

# Hierarchical RNN layer 
def param_init_rnn_hiero(options, params, prefix='rnn_hiero', nin=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dimctx == None:
        dimctx = options['dim']
    dim = dimctx

    params = param_init_rnn(options, params, prefix, nin=nin, dim=dim)

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    # stop probability:
    W_st = norm_weight(dim, 1)
    params[_p(prefix,'W_st')] = W_st
    b_st = numpy.zeros((1,)).astype('float32')
    params[_p(prefix,'b_st')] = b_st

    return params

def rnn_hiero_layer(tparams, context, options, prefix='rnn_hiero', 
                    context_mask=None, **kwargs):

    nsteps = context.shape[0]
    if context.ndim == 3:
        n_samples = context.shape[1]
    else:
        n_samples = 1

    # mask
    if context_mask == None:
        mask = tensor.alloc(1., context.shape[0], 1)
    else:
        mask = context_mask

    dim = tparams[_p(prefix, 'Ux')].shape[0]

    # initial/previous state
    init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix,'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, h_, ctx_, alpha_, v_, pctx_,
              Wd_att, U_att, c_tt, Ux, Wx, bx, W_st, b_st):

        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pctx_ + pstate_[None,:,:] 
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:,:,None]).sum(0) # current context

        preactx = tensor.dot(h_, Ux)
        preactx += tensor.dot(ctx_, Wx)
        preactx += bx

        h = tensor.tanh(preactx)

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        # compute stopping probability
        ss = tensor.nnet.sigmoid(tensor.dot(h, W_st) + b_st)
        v_ = v_ * (1. - ss)[:,0][:,None]

        return h, ctx_, alpha.T, v_[:,0] #, pstate_, preact, preactx, r, u

    rval, updates = theano.scan(_step, 
                                sequences=[mask],
                                outputs_info = [init_state, 
                                                tensor.alloc(0., n_samples, context.shape[2]),
                                                tensor.alloc(0., n_samples, context.shape[0]),
                                                tensor.alloc(1., n_samples)],
                                                #None, None, None, 
                                                #None, None],
                                non_sequences=[pctx_,
                                               tparams[_p(prefix,'Wd_att')],
                                               tparams[_p(prefix,'U_att')],
                                               tparams[_p(prefix, 'c_tt')],
                                               tparams[_p(prefix, 'Ux')],
                                               tparams[_p(prefix, 'Wx')],
                                               tparams[_p(prefix, 'bx')],
                                               tparams[_p(prefix, 'W_st')],
                                               tparams[_p(prefix, 'b_st')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)

    rval[0] = rval[0] * rval[3][:,:,None]
    return rval

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def param_init_gru_nonlin(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    
    U_nl = numpy.concatenate([ortho_weight(dim),
                              ortho_weight(dim)], axis=1)
    params[_p(prefix,'U_nl')] = U_nl
    params[_p(prefix,'b_nl')] = numpy.zeros((2 * dim,)).astype('float32')

    Ux_nl = ortho_weight(dim)
    params[_p(prefix,'Ux_nl')] = Ux_nl
    params[_p(prefix,'bx_nl')] = numpy.zeros((dim,)).astype('float32')
    
    return params

def gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step, 
                                sequences=seqs,
                                outputs_info = [tensor.alloc(0., n_samples, dim)],
                                                #None, None, None, None],
                                non_sequences = [tparams[_p(prefix, 'U')], 
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval

# Conditional GRU layer without Attention
def param_init_gru_cond_simple(options, params, prefix='gru_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']


    params = param_init_gru(options, params, prefix, nin=nin, dim=dim)


    # context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[_p(prefix,'Wc')] = Wc

    Wcx = norm_weight(dimctx,dim)
    params[_p(prefix,'Wcx')] = Wcx

    return params

def gru_cond_simple_layer(tparams, state_below, options, prefix='gru', 
                          mask=None, context=None, one_step=False, 
                          init_memory=None, init_state=None, 
                          context_mask=None,
                          **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 2, 'Context must be 2-d: #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc')])
    pctxx_ = tensor.dot(context, tparams[_p(prefix,'Wcx')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, pctx_, pctxx_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_
        preact += pctx_
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += pctxx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]] 

    if one_step:
        rval = _step(*(seqs+[init_state, pctx_, pctxx_]+shared_vars))
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info=[init_state], 
                                    non_sequences=[pctx_,
                                                   pctxx_]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval

# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']

    params = param_init_gru_nonlin(options, params, prefix, nin=nin, dim=dim)

    # context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[_p(prefix,'Wc')] = Wc

    Wcx = norm_weight(dimctx,dim)
    params[_p(prefix,'Wcx')] = Wcx
    
    

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim,dimctx)
    params[_p(prefix,'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params

def gru_cond_layer(tparams, state_below, options, prefix='gru', 
                    mask=None, context=None, one_step=False, 
                    init_memory=None, init_state=None, 
                    context_mask=None,
                    **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix,'b_att')]
        
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    #state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])
    #import ipdb; ipdb.set_trace()
    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:,None] * h1 + (1. - m_)[:,None] * h_
        
        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None,:,:] 
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:,:,None]).sum(0) # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:,None] * h2 + (1. - m_)[:,None] * h1

        return h2, ctx_, alpha.T #, pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix,'W_comb_att')],
                   tparams[_p(prefix,'U_att')], 
                   tparams[_p(prefix, 'c_tt')], 
                   tparams[_p(prefix, 'Ux')], 
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state, 
                                                    tensor.alloc(0., n_samples, context.shape[2]),
                                                    tensor.alloc(0., n_samples, context.shape[0])],
                                                    #None, None, None, 
                                                    #None, None],
                                    non_sequences=[pctx_,
                                                   context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# Hierarchical GRU layer 
def param_init_gru_hiero(options, params, prefix='gru_hiero', nin=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dimctx == None:
        dimctx = options['dim']
    dim = dimctx

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim, hiero=True)

    # context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[_p(prefix,'Wc')] = Wc

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    # stop probability:
    W_st = norm_weight(dim, 1)
    params[_p(prefix,'W_st')] = W_st
    b_st = -0. * numpy.ones((1,)).astype('float32')
    params[_p(prefix,'b_st')] = b_st

    return params

def gru_hiero_layer(tparams, context, options, prefix='gru_hiero', 
                    context_mask=None, **kwargs):

    nsteps = context.shape[0]
    if context.ndim == 3:
        n_samples = context.shape[1]
    else:
        n_samples = 1

    # mask
    if context_mask == None:
        mask = tensor.alloc(1., context.shape[0], 1)
    else:
        mask = context_mask

    dim = tparams[_p(prefix, 'W_st')].shape[0]

    # initial/previous state
    init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix,'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step_slice(m_, h_, ctx_, alpha_, v_, pp_, cc_,
                    U, Wc, Wd_att, U_att, c_tt, Ux, Wx, bx, W_st, b_st):
        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pp_ + pstate_[None,:,:] 
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx = (cc_ * alpha[:,:,None]).sum(0) # current context

        preact = tensor.dot(h_, U)
        preact += tensor.dot(ctx, Wc)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx += tensor.dot(ctx, Wx)
        preactx += bx

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        # compute stopping probability
        ss = tensor.nnet.sigmoid(tensor.dot(h, W_st) + b_st)
        v_ = v_ * (1. - ss)[:,0][:,None]

        return h, ctx, alpha.T, v_[:,0] #, pstate_, preact, preactx, r, u

    _step = _step_slice

    rval, updates = theano.scan(_step, 
                                sequences=[mask],
                                outputs_info = [init_state, 
                                                tensor.alloc(0., n_samples, context.shape[2]),
                                                tensor.alloc(0., n_samples, context.shape[0]),
                                                tensor.alloc(1., n_samples)],
                                                #None, None, None, 
                                                #None, None],
                                non_sequences=[pctx_, 
                                               context,
                                               tparams[_p(prefix, 'U')],
                                               tparams[_p(prefix, 'Wc')],
                                               tparams[_p(prefix,'Wd_att')], 
                                               tparams[_p(prefix,'U_att')], 
                                               tparams[_p(prefix, 'c_tt')], 
                                               tparams[_p(prefix, 'Ux')], 
                                               tparams[_p(prefix, 'Wx')],
                                               tparams[_p(prefix, 'bx')], 
                                               tparams[_p(prefix, 'W_st')], 
                                               tparams[_p(prefix, 'b_st')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)

    rval[0] = rval[0] * rval[3][:,:,None]
    return rval

# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'U')].shape[0]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, c, i, f, o, preact

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_below],
                                outputs_info = [tensor.alloc(0., n_samples, dim),
                                                tensor.alloc(0., n_samples, dim),
                                                None, None, None, None],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']

    params = param_init_lstm(options, params, prefix, nin, dim)

    # context to LSTM
    Wc = norm_weight(dimctx,dim*4)
    params[_p(prefix,'Wc')] = Wc

    # attention: prev -> hidden
    Wi_att = norm_weight(nin,dimctx)
    params[_p(prefix,'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params

def lstm_cond_layer(tparams, state_below, options, prefix='lstm', 
                    mask=None, context=None, one_step=False, 
                    init_memory=None, init_state=None, 
                    context_mask=None,
                    **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_memory, 'previous memory must be provided'
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory 
    if init_memory == None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix,'b_att')]

    # projected x
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, xc_, h_, c_, ctx_, alpha_, pctx_):

        # attention
        pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')])
        pctx__ = pctx_ + pstate_[None,:,:] 
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, tparams[_p(prefix,'U_att')])+tparams[_p(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:,:,None]).sum(0) # current context

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tensor.dot(ctx_, tparams[_p(prefix, 'Wc')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, c, ctx_, alpha.T, pstate_, preact, i, f, o

    if one_step:
        rval = _step(mask, state_below, state_belowc, init_state, init_memory, None, None, pctx_)
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=[mask, state_below, state_belowc],
                                    outputs_info = [init_state, init_memory,
                                                    tensor.alloc(0., n_samples, context.shape[2]),
                                                    tensor.alloc(0., n_samples, context.shape[0]),
                                                    None, None, None, 
                                                    None, None],
                                    non_sequences=[pctx_],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile)
    return rval


# initialize all parameters
def init_params(options):
    numpy.random.seed(1234)
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

   
    # encoder: LSTM
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder', 
                                              nin=options['dim_word'], dim=options['dim'])
    ctxdim = options['dim']
    if not options['decoder'].endswith('simple'):
        ctxdim = options['dim'] * 2
        params = get_layer(options['encoder'])[0](options, params, prefix='encoder_r', 
                                                  nin=options['dim_word'], dim=options['dim'])
        if options['hiero']:
            params = get_layer(options['hiero'])[0](options, params, prefix='hiero', 
                                                    nin=2*options['dim'], dimctx=2*options['dim'])
    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctxdim, nout=options['dim'])
    if options['encoder'] == 'lstm':
        params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctxdim, nout=options['dim'])
    # decoder: LSTM
    params = get_layer(options['decoder'])[0](options, params, prefix='decoder', 
                                              nin=options['dim_word'], dim=options['dim'], 
                                              dimctx=ctxdim)

    

    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'], ortho=False)
    params = get_layer('ff_nb')[0](options, params, prefix='ff_nb_logit_prev', nin=options['dim_word'], nout=options['dim_word'], ortho=False)
    params = get_layer('ff_nb')[0](options, params, prefix='ff_nb_logit_ctx', nin=ctxdim, nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])

    return params

# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]
    src_lengths = x_mask.sum(axis=0)
    
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    if options['decoder'].endswith('simple'):
        ctx = proj[0][-1]
        ctx_mean = ctx
    else:
        embr = tparams['Wemb'][xr.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
        projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                                 prefix='encoder_r',
                                                 mask=xr_mask)
        ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        if options['hiero']:
            #ctx = tensor.dot(ctx, tparams['W_hiero'])
            rval = get_layer(options['hiero'])[1](tparams, ctx, options,
                                                  prefix='hiero',
                                                  context_mask=x_mask)
            ctx = rval[0]
            opt_ret['hiero_alphas'] = rval[2]
            opt_ret['hiero_betas'] = rval[3]
        # initial state/cell
        # ctx_mean = ctx.mean(0)
        # ctx_mean = (ctx * x_mask[:,:,None]).sum(0) / x_mask.sum(0)[:,None]
        ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = None
    if options['encoder'] == 'lstm':
        init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')
    # word embedding (target)
    emb = tparams['Wemb_dec'][y.flatten()].reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    

    # decoder
    proj = get_layer(options['decoder'])[1](tparams, emb, options, 
                                            prefix='decoder', 
                                            mask=y_mask, context=ctx, 
                                            context_mask=x_mask,
                                            one_step=False, 
                                            init_state=init_state,
                                            init_memory=init_memory)
    proj_h = proj[0]
    if options['decoder'].endswith('simple'):
        ctxs = ctx[None,:,:]
    else:
        if options['decoder'].startswith('lstm'):
            ctxs = proj[2]
            opt_ret['dec_alphas'] = proj[3]
        else:
            ctxs = proj[1]
            opt_ret['dec_alphas'] = proj[2]
    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff_nb')[1](tparams, emb, options, prefix='ff_nb_logit_prev', activ='linear')
    logit_ctx = get_layer('ff_nb')[1](tparams, ctxs, options, prefix='ff_nb_logit_ctx', activ='linear')
    
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))
    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0],y.shape[1]])
    cost = (cost * y_mask).sum(0)
    
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost

# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])


    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options, prefix='encoder')
    if options['decoder'].endswith('simple'):
        ctx = proj[0][-1]
        ctx_mean = ctx
    else:
        projr = get_layer(options['encoder'])[1](tparams, embr, options, prefix='encoder_r')
        ctx = concatenate([proj[0],projr[0][::-1]], axis=proj[0].ndim-1)
        if options['hiero']:
            rval = get_layer(options['hiero'])[1](tparams, ctx, options, prefix='hiero')
            ctx = rval[0]
        # initial state/cell
        # ctx_mean = ctx.mean(0)
        ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    if options['encoder'] == 'lstm':
        init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    if options['decoder'].startswith('lstm'):
        outs += [init_memory]

    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    
    init_state = tensor.matrix('init_state', dtype='float32')
    if options['decoder'].startswith('lstm'):
        init_memory = tensor.matrix('init_memory', dtype='float32')
    else:
        init_memory = None
    
    n_timesteps = ctx.shape[0]
        
    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:,None] < 0, tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]), 
                        tparams['Wemb_dec'][y])

    

    proj = get_layer(options['decoder'])[1](tparams, emb, options, 
                                            prefix='decoder', 
                                            mask=None, context=ctx, 
                                            one_step=True, 
                                            init_state=init_state,
                                            init_memory=init_memory)
    if options['decoder'].endswith('simple'):
        next_state = proj
        ctxs = ctx
    else:
        next_state = proj[0]
        ctxs = proj[1]
        if options['decoder'].startswith('lstm'):
            next_memory = proj[1]
            ctxs = proj[2]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options, prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff_nb')[1](tparams, emb, options, prefix='ff_nb_logit_prev', activ='linear')
    logit_ctx = get_layer('ff_nb')[1](tparams, ctxs, options, prefix='ff_nb_logit_ctx', activ='linear')
    
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..', 
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    if options['decoder'].startswith('lstm'):
        inps += [init_memory]
        outs += [next_memory]
    
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next

# generate sample
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30, 
               minlen=-1, stochastic=True, argmax=False):
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if options['decoder'].startswith('lstm'):
        hyp_memories = []
    
    ret = f_init(x)
    next_state = ret.pop(0)
    ctx0 = ret.pop(0)
    if options['decoder'].startswith('lstm'):
        next_memory = ret.pop(0)
    
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        if options['decoder'].endswith('simple'):
            ctx = numpy.tile(ctx0, [live_k, 1])
        else:
            ctx = numpy.tile(ctx0.reshape((ctx0.shape[0],ctx0.shape[2])), 
                                          [live_k, 1, 1]).transpose((1,0,2))
        inps = [next_w, ctx, next_state]
        if options['decoder'].startswith('lstm'):
            inps += [next_memory]
        
        ret = f_next(*inps)
        next_p = ret.pop(0)
        next_w = ret.pop(0)
        next_state = ret.pop(0)
        if options['decoder'].startswith('lstm'):
            next_memory = ret.pop(0)

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0,nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            
            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            if options['decoder'].startswith('lstm'):
                new_hyp_memories = []
            
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                if options['decoder'].startswith('lstm'):
                    new_hyp_memories.append(copy.copy(next_memory[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            if options['decoder'].startswith('lstm'):
                hyp_memories = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    if len(new_hyp_samples[idx]) >= minlen:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    if options['decoder'].startswith('lstm'):
                        hyp_memories.append(new_hyp_memories[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            if options['decoder'].startswith('lstm'):
                next_memory = numpy.array(hyp_memories)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])


    return sample, sample_score

def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    iterator.start()
    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=50, n_words_src=options['n_words_src'], n_words=options['n_words'])
        
        if x == None:
            continue

        pprobs = f_log_probs(x,x_mask,y,y_mask)
        for pp in pprobs:
            probs.append(pp)

        if verbose:
            print >>sys.stderr, '%d samples computed'%(n_done)

    return numpy.array(probs)

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost):
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    rg2_new = [0.95 * rg2 + 0.05 * (g ** 2) for rg2, g in zip(running_grads2, grads)]
    rg2up = [(rg2, r_n) for rg2, r_n in zip(running_grads2, rg2_new)]
    
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(grads, running_up2, rg2_new)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    inp += [lr]
    f_update = theano.function(inp, cost, updates=rg2up+ru2up+param_up, on_unused_input='ignore', profile=profile)

    return f_update

def debugging_adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=profile)
    
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100, # word vector dimensionality
          dim=1000, # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          hiero=None, #'gru_hiero', # or None
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0., 
          alpha_c=0., 
          diag_c=0.,
          lrate=0.01, 
          n_words_src=100000,
          n_words=100000,
          maxlen=100, # maximum length of the description
          optimizer='rmsprop', 
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000, # save the parameters after every saveFreq updates
          sampleFreq=100, # generate some samples after every sampleFreq updates
          dataset='wmt14enfr',
          dictionary=None, # word dictionary
          dictionary_src=None, # word dictionary
          use_dropout=False,
          reload_=False,
          correlation_coeff=0.1,
          clip_c=0.):

    # Model options
    model_options = locals().copy()
    
    if dictionary:
        with open(dictionary, 'rb') as f:
            word_dict = pkl.load(f)
        word_idict = dict()
        for kk, vv in word_dict.iteritems():
            word_idict[vv] = kk

    if dictionary_src:
        with open(dictionary_src, 'rb') as f:
            word_dict_src = pkl.load(f)
        word_idict_src = dict()
        for kk, vv in word_dict_src.iteritems():
            word_idict_src[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)
    #import ipdb; ipdb.set_trace()
    print 'Loading data'
    load_data, prepare_data = get_dataset(dataset)
    train, valid, test = load_data(batch_size=batch_size)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
          x, x_mask, y, y_mask, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    #theano.printing.debugprint(cost.mean(), file=open('cost.txt', 'w'))

    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:,None]-
                                opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    if model_options['hiero'] != None:
        print 'Building f_beta...',
        f_beta = theano.function([x, x_mask], opt_ret['hiero_betas'], profile=profile)
        print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'
    print 'Building f_grad...',
    f_grad = theano.function(inps, grads, profile=profile)
    print 'Done'

    #Cliping gradients
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    #f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0
        #import ipdb; ipdb.set_trace()
        train.start()
        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen, 
                                                n_words_src=n_words_src, n_words=n_words)

            if x == None:
                #print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()
            #cost = f_grad_shared(x, x_mask, y, y_mask)
            #f_update(lrate)
            cost = f_update(x, x_mask, y, y_mask, lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                #import ipdb; ipdb.set_trace()

                #if best_p != None:
                #    params = best_p
                #else:
                params = unzip(tparams)

                saveto_list = saveto.split('/')
                saveto_list[-1] = 'epoch' + str(eidx) + '_' + 'nbUpd' + str(uidx) + '_' + saveto_list[-1]
                saveName = '/'.join(saveto_list)
                numpy.savez(saveName, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl'%saveName, 'wb'))
                print 'Done'

            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5,x.shape[1])):
                    stochastic = False
                    sample, score = gen_sample(tparams, f_init, f_next, x[:,jj][:,None], 
                                               model_options, trng=trng, k=1, maxlen=30, 
                                               stochastic=stochastic, argmax=True)
                    print 'Source ',jj,': ',
                    for vv in x[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict_src:
                            print word_idict_src[vv], 
                        else:
                            print 'UNK',
                    print
                    print 'Truth ',jj,' : ',
                    for vv in y[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv], 
                        else:
                            print 'UNK',
                    print
                    if model_options['hiero']:
                        betas = f_beta(x[:,jj][:,None], x_mask[:,jj][:,None])
                        print 'Validity ', jj,': ',
                        for vv,bb in zip(y[:,jj],betas[:,0]):
                            if vv == 0:
                                break
                            print bb,
                        print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv], 
                        else:
                            print 'UNK',
                    print

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0
                #for _, tindex in kf:
                #    x, mask = prepare_data(train[0][train_index])
                #    train_err += (f_pred(x, mask) == train[1][tindex]).sum()
                #train_err = 1. - numpy.float32(train_err) / train[0].shape[0]

                #train_err = pred_error(f_pred, prepare_data, train, kf)
                if valid != None:
                    valid_err = pred_probs(f_log_probs, prepare_data, model_options, valid).mean()
                if test != None:
                    test_err = pred_probs(f_log_probs, prepare_data, model_options, test).mean()

                history_errs.append([valid_err, test_err])

                if uidx == 0 or valid_err <= numpy.array(history_errs)[:,0].min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience,0].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

                print 'Seen %d samples'%n_samples

        #print 'Epoch ', eidx, 'Update ', uidx, 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        #print 'Seen %d samples'%n_samples

        if estop:
            break

    if best_p is not None: 
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0
    #train_err = pred_error(f_pred, prepare_data, train, kf)
    if valid != None:
        valid_err = pred_probs(f_log_probs, prepare_data, model_options, valid).mean()
    if test != None:
        test_err = pred_probs(f_log_probs, prepare_data, model_options, test).mean()


    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    if best_p != None:
        params = copy.copy(best_p)
    else:
        params = unzip(tparams)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err, 
                valid_err=valid_err, test_err=test_err, history_errs=history_errs, 
                **params)

    return train_err, valid_err, test_err



if __name__ == '__main__':
    pass












    


