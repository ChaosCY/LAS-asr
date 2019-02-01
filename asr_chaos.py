import os
import librosa
import argparse
import numpy as np
import tensorflow as tf

import utils
from data_corpus import Corpus
from las_model import LAS
from evaluation import get_edit_distance



#######################################################################################
# parameters
#######################################################################################
WAV_FILE = ['data/wav']
TRN_FILE = ['data/trn']
OUTPUT_MFCC_FILE = ['data/mfcc']
TEST_WAV_FILE = ['data/wav']
TEST_TRN_FILE = ['data/trn']
TEST_OUTPUT_MFCC_FILE = ['data/mfcc']
# the bias word in inference stage
INFERENCE_BIAS = [['shen', 'fen', 'zheng'], 
                  ['lian', 'xi', 'fang', 'shi'], 
                  ['dian', 'hua', 'hao'],
                  ['yi'],
                  ['er'],
                  ['san'],
                  ['si'],
                  ['wu'],
                  ['liu'],
                  ['qi'],
                  ['ba'],
                  ['jiu'],
                  ['ling']]

def iter_epoches(sess, epoch, data_corpus_instance, model):
    ''' iterate epoches 
    Args:
    sess: session
    epoch: current epoch
    data_corpus_instance: instance of data_corpus
    model: model graph
    '''
    batches = data_corpus_instance.batch_generator()
    losses = []
    while True:
        try:
            mfcc_features, audio_seq_len, labels, label_seq_len, biases, bias_seq_len \
                                                                  = get_feeds(batches)
            bias_att_len = [len(biases) for _ in range(len(labels))]
            feed = {model.audios: mfcc_features, model.char_ids: labels, 
                    model.bias_ids: biases,
                    model.audio_sequence_lengths: audio_seq_len, 
                    model.char_sequence_lengths: label_seq_len,
                    model.bias_sequence_lengths: bias_seq_len,
                    model.bias_attention_lengths: bias_att_len}
            train_ops = [model.out_logits, model.pred, model.train_op, model.train_loss]
            logits, preds, _, loss_batch = run_train_op(sess, train_ops, feed)
            print("epoches: %3d, loss: %.6f" % (epoch, loss_batch))
            losses.append(loss_batch)
        except StopIteration:
            # if arrive at the file end, break current epoch
            break
    return np.mean(losses)

def run_train_op(sess, ops, feed):
    # start the operation
    return sess.run(ops, feed)
    
def get_feeds(batch_generator):
    # return feeds
    mfcc_features, audio_seq_len, labels, label_seq_len, biases, bias_seq_len \
                                                      = next(batch_generator)
    mfcc_features = np.transpose(mfcc_features, [0, 2, 1])
    return mfcc_features, audio_seq_len, labels, label_seq_len, biases, bias_seq_len

def main():
    # setting parameters
    parser = argparse.ArgumentParser(
              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default=None,
              help='TRAIN or FINETUNE or INFER.')
    parser.add_argument('--epoches', type=int, default=1000,
              help='num epoches.')
    parser.add_argument('--batch_size', type=int, default=50,
              help='minibatch size.')
    parser.add_argument('--batch_increase', type=bool, default=True,
              help='whether to increase the batch_size')
    parser.add_argument('--num_layers_encoder', type=int, default=2,
              help='number of encoder layers.')
    parser.add_argument('--num_layers_decoder', type=int, default=1,
              help='number of decoder layers.')
    parser.add_argument('--embedding_dim', type=int, default=100,
              help='dimension of the embedding vectors in the embedding matrix.')
    parser.add_argument('--num_heads', type=int, default=8,
              help='number of head in multi_heads attention.')
    parser.add_argument('--rnn_size_encoder', type=int, default=256,
              help='number of hidden units in encoder.')
    parser.add_argument('--rnn_size_decoder', type=int, default=256,
              help='number of hidden units in decoder.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
              help='learning rate in every training step.')
    parser.add_argument('--learning_rate_decay', type=float, default=1,
              help='only if exponential learning rate is used.')
    parser.add_argument('--learning_rate_decay_steps', type=int, default=100,
              help='learning rate decay period.')
    parser.add_argument('--max_lr', type=float, default=0.01,
              help='only if cyclic learning rate is used.')
    parser.add_argument('--label_smoothing', type=float, default=0,
              help='the label smoothing rate.')
    parser.add_argument('--keep_probability_i', type=float, default=1,#0.825
              help='values inspired by Jeremy Howard\'s fast.ai course.')
    parser.add_argument('--keep_probability_o', type=float, default=1,#0.895
              help='values inspired by Jeremy Howard\'s fast.ai course.')
    parser.add_argument('--keep_probability_h', type=float, default=1,#0.86
              help='values inspired by Jeremy Howard\'s fast.ai course.')
    parser.add_argument('--keep_probability_e', type=float, default=1,#0.986
              help='values inspired by Jeremy Howard\'s fast.ai course.')
    # A bug occurred when 0 choosed. Please set beam_width greater than 0 
    # at infer stage before the problem is resolved.
    parser.add_argument('--beam_width', type=int, default=1,
              help='only used in inference, for Beam Search.')
    parser.add_argument('--clip', type=int, default=5,
              help='value to clip the gradients to in training process.')
    parser.add_argument('--inference_targets', type=int, default=False,
              help='maximum iterations at decoding period')
    parser.add_argument('--use_cyclic_lr', type=int, default=False,
              help='use cyclical learning rates.')
    parser.add_argument('--key_words_biasing', type=bool, default=True,
              help='whether implement the CLAS for key words, default YES')
    parser.add_argument('--attention_type', type=str, default='MultiHeadAttention',
              help='MultiHeadAttention or BahdanauAttention can be selected.')
    parser.add_argument('--attention_type_bias', type=str, default='MultiHeadAttention',
              help='MultiHeadAttention or BahdanauAttention can be selected.')
    parser.add_argument('--crf_layer', type=bool, default=True,
              help='if add a crf layer on the decoder outputs.')
    parser.add_argument('--dev', type=str, default='cpu',
              help='training by CPU or GPU, input cpu or gpu:0 or gpu:1 or gpu:2 or gpu:3.')
    args = parser.parse_args()
    
    ##################################################################################
    # initital the data, model graph, parameters
    ##################################################################################
    print("creating data operator...")
    # param vocab_create_mode='BUILD' in the first training
    # the trn files and wav files saved in different folders
    if args.mode == 'INFER':
        args.batch_size = 1
        data = Corpus(trn_file=TEST_TRN_FILE, wav_file=TEST_WAV_FILE, \
                      mfcc_file=TEST_OUTPUT_MFCC_FILE, args=args, \
                      vocab_create_mode='LOAD', mfcc_create='N')
    else:
        data = Corpus(trn_file=TRN_FILE, wav_file=WAV_FILE, mfcc_file=OUTPUT_MFCC_FILE, \
                  args=args, vocab_create_mode='LOAD', mfcc_create='N')
    print("building model graph...")
    model = LAS(args, data.vocab)
    model.build_model()
    saver = tf.train.Saver()
    
    sess = tf.Session()
    print("initializing parameters...")
    sess.run(tf.global_variables_initializer())
    
    ##################################################################################
    # TRAIN or INFERENCE stage
    ##################################################################################
    if args.mode=='TRAIN':
        ## train 
        with tf.device("/" + str(args.dev)):
            best_loss = np.inf
            for epoch in range(args.epoches):
              ## """attempt to increase the batch_size, increase 10 when the epoches increase 50, 
              ## but the max batch_size should be 100 because of the memory limit."""
              if epoch%50==0 and args.batch_increase and (epoch !=0):
                  args.batch_size+=10
              if args.batch_size>=100:
                  args.batch_increase=False
              
              avg_loss = iter_epoches(sess, epoch, data, model)
              # if current loss is smaller than the best
              if avg_loss<best_loss:
                  best_loss = avg_loss
                  print("best_loss: %6f" % (best_loss))
                  # save model
                  save_path = saver.save(sess, "save/model.ckpt")
                  
    elif args.mode=='FINETUNE':
        ## train the model base on the parameters of the previous training
        with tf.device("/" + str(args.dev)):
            # read model from file
            saver.restore(sess, "save/model.ckpt")
            best_loss = np.inf
            for epoch in range(args.epoches):
                avg_loss = iter_epoches(sess, epoch, data, model)
                if avg_loss<best_loss:
                    best_loss = avg_loss
                    # save model
                    print("best_loss: %6f" % (best_loss))
                    save_path = saver.save(sess, "save/model.ckpt")
                    
    elif args.mode=='INFER':
        with tf.device("/" + str(args.dev)):
            # read model parameters from file
            saver.restore(sess, "save/model.ckpt")
            batches = data.batch_generator()
            lines = []
            wers = []
            count = 0
            biases = INFERENCE_BIAS
            bias_seq_len = [len(bias) for bias in biases]
            biases = data.trans_label_to_index(biases)
            biases = data.padding(biases, bias_seq_len)
            while True:
                count += 1
                if count%1==0:
                    print(str(count) + ' finished...')
                try:
                    mfcc_features, audio_seq_len, labels, label_seq_len, _, _ = \
                                                                get_feeds(batches)
                    bias_att_len = [len(biases) for _ in range(len(labels))]
                    feed = {model.audios: mfcc_features, 
                            model.audio_sequence_lengths: audio_seq_len,
                            model.bias_ids: biases,
                            model.char_sequence_lengths: label_seq_len,
                            model.bias_sequence_lengths: bias_seq_len,
                            model.bias_attention_lengths: bias_att_len}
                    train_ops = model.sample_words
                    preds = run_train_op(sess, train_ops, feed)
                    for p,label in zip(preds,labels):
                        sen = np.transpose(np.array(p), [1, 0])
                        line = ' '.join(data.trans_index_to_label(list(sen[0])))
                        lines.append(line)
                        # calculate the WER
                        wers.append(get_edit_distance(line, label))
                except StopIteration:
                    break
            wer = np.mean(np.array(wers))
            print(wer)
            utils.write_list_to_file('pred/predictions.txt', lines, 'a+')

if __name__ == '__main__':
    main()