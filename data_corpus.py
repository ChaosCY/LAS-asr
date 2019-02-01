import os
import librosa
import random
import numpy as np
import tensorflow as tf
from python_speech_features import mfcc

import utils

class Corpus(object):
    def __init__(self, trn_file, wav_file, mfcc_file, args, 
                vocab_create_mode='BUILD', mfcc_create='Y'):
        ''' 
        Args:
        data_file: data file path
        vocab_create_mode: 
                BUILD: create the vocab dict from raw label data
                LOAD : read from file directly
        '''
        self.args = args

        #trn file path 
        self.trn_file = trn_file
        #wav file path
        self.wav_file = wav_file
        #mfcc file path
        self.mfcc_file = mfcc_file

        # data file path
        #self.data_file = data_file
        # <EOS>: end of the sentenset tag
        # <SOS>: start of the sentenset tag
        # <PAD>: padding tag
        self.special_signs = ['<EOS>', '<SOS>', '<PAD>', '<BIAS>']
        # label to index dict
        self.vocab = {}
        # index to label dict
        self.inverse_vocab = {}

        if vocab_create_mode=='BUILD':
            self.label_process()
        elif vocab_create_mode=='LOAD':
            self.vocab = utils.load_from_pkl('vocab.pkl')
            self.inverse_vocab = utils.invert_dict(self.vocab)

        if mfcc_create=='Y':
            for i in range(len(self.wav_file)):
                wavlist = os.listdir(self.wav_file[i])
                for j in range(len(wavlist)):
                    wav_path = os.path.join(self.wav_file[i], wavlist[j])
                    # invert the radio to the mfcc feature
                    mfcc = self.read_wav_file(wav_path, 26, 9)
                    mfcc = np.transpose(mfcc)
                    np.save(os.path.join(self.mfcc_file[i], \
                            os.path.splitext(wavlist[j])[0]), mfcc, 'utf-8')

    def read_wav_file(self, file_path, numcep, numcontext=None):
        ''' read wav files
        '''
        wave, sr = librosa.load(file_path, mono=True)
        # get the mfcc features
        # features = librosa.feature.mfcc(wave, sr)
        features = mfcc(wave, samplerate=sr, numcep=numcep, nfft=551)
        features = features[::2]
        # One stride per time step in the input
        num_strides = len(features)

        # add a zero context
        empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        window_size = 2 * numcontext + 1
        train_inputs = np.lib.stride_tricks.as_strided(
                      features,
                      (num_strides, window_size, numcep),
                      (features.strides[0], features.strides[0], features.strides[1]),
                      writeable=False)
        # Flatten the second and third dimensions
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        # Whiten inputs (TODO: Should we whiten?)
        # Copy the strided array so that we can write to it safely
        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

        return train_inputs


    def load_array_from_file(self, path):
        return np.load(path)

    def load_label_file(self, path, mode):
        ''' read target label files
        Args:
            path: file path
            mode: 
                CHAR: split in character units
                WORD: split in word units      (not implement yet)
        '''
        label = []
        with open(path, 'r', encoding='utf-8') as fin:
            i = 0
            for line in fin:
                if i==1:
                    if mode=='CHAR':
                        #label = list(line)
                        #label = [ch if ch != ' ' else '<SPACE>' for ch in label]
                        label = line.strip().split(' ')
                    elif mode=='WORD':
                        # TODO: implement the word level prediction
                        #label = line.strip().split(' ')
                        pass
                #break
                i += 1
        return label


    def create_lookup_dict(self, label_set):
        # label-index lookup mapping
        return dict(zip(label_set, range(len(label_set))))

    def trans_index_to_label(self, indexs):
        ''' invert a label index to the label
        '''
        indexs = [self.inverse_vocab[i] for i in indexs if i!=-1]
        return indexs

    def trans_label_to_index(self, labels):
        ''' invert a label to the index
        '''
        labels = [[self.vocab[l] for l in one_label] for one_label in labels]
        return labels

    def label_process(self):
        ''' create the vocab dict and write to a pickle file
        '''
        label_set = set(self.special_signs)
        for trn_file_path in self.trn_file:
          files = os.listdir(trn_file_path)
          for file in files:
              if not file.endswith(".wav") and not file.endswith(".mp3"):
                  file_path = os.path.join(trn_file_path, file)
                  label = self.load_label_file(file_path, 'CHAR')
                  label_set |= set(label)
        
        self.vocab = self.create_lookup_dict(label_set)
        utils.write_to_pkl('vocab.pkl', self.vocab)

    def padding(self, seq_list, seq_len):
        seq_list = [np.pad(s, (0, max(seq_len)-len(s)), 'constant') for s in seq_list]
        return seq_list
      
    def add_bias(self, label, biases, bias_seq_len):
        start = random.randint(1,len(label)-1)
        end = random.randint(1,len(label)-1)
        if end-start > 5:
            end = start+5
        line = ' '.join(label)
        if start < end:
            bias = label[start:end]
            if bias not in biases:
                biases.append(bias)
                bias_seq_len.append(len(bias))
        for bias in biases:
            b_line = ' '.join(bias)
            if b_line in line:
                b_index = line.index(b_line)
                line = line[:b_index+len(b_line)]+' <BIAS>'+line[b_index+len(b_line):]
                label = line.strip().split(' ')
    
    def batch_generator(self):
        ''' batch generator
        '''
        #files = os.listdir(self.data_file)
        #files.sort()
        # mfcc features
        mfcc_features = []
        # target labels
        labels = []
        # audio sequence lengths
        mfcc_seq_len = []
        # label sequence lengths
        label_seq_len = []
        # bias 
        biases = []
        # bias sequence lengths
        bias_seq_len = []

        """trn"""
        filelist_trn = os.listdir(self.trn_file[0])
        filelist_trn.sort()
        """mfcc"""
        filelist_mfcc = os.listdir(self.mfcc_file[0])
        filelist_mfcc.sort()

        labels_length = len(filelist_trn)
        indexes = np.arange(labels_length)
        np.random.shuffle(indexes)

        filelist_trn = list(np.array(filelist_trn)[indexes])
        filelist_mfcc = list(np.array(filelist_mfcc)[indexes])
        
        if len(self.mfcc_file) > 1:
            """trn"""
            filelist_trn_extend = os.listdir(self.trn_file[1])
            filelist_trn_extend.sort()
            """mfcc"""
            filelist_mfcc_extend = os.listdir(self.mfcc_file[1])
            filelist_mfcc_extend.sort()
            
            labels_length = len(filelist_trn_extend)
            indexes = np.arange(labels_length)
            np.random.shuffle(indexes)

            filelist_trn_extend = list(np.array(filelist_trn_extend)[indexes])
            filelist_mfcc_extend = list(np.array(filelist_mfcc_extend)[indexes])

        file_append = (self.args.batch_size - (len(filelist_trn) % self.args.batch_size)) \
                                                                  % self.args.batch_size
        for i in range(file_append):
            filelist_trn.append(filelist_trn[i])
            filelist_mfcc.append(filelist_mfcc[i])

        for i in range(len(filelist_trn)):
            file_path = os.path.join(self.trn_file[0], filelist_trn[i])
            label = self.load_label_file(file_path, 'CHAR')
            label.append('<EOS>')
            label.insert(0, '<SOS>')
            # add bias
            self.add_bias(label, biases, bias_seq_len)
            label_seq_len.append(len(label))
            labels.append(label)

            file_path = os.path.join(self.mfcc_file[0], filelist_mfcc[i])
            # invert the radio to the mfcc feature
            mfcc = self.load_array_from_file(file_path)
            mfcc_seq_len.append(mfcc.shape[1])
            mfcc_features.append(mfcc)
            
            if len(self.mfcc_file) > 1:
                file_path = os.path.join(self.trn_file[1], filelist_trn_extend[i])
                label = self.load_label_file(file_path, 'CHAR')
                label.append('<EOS>')
                label.insert(0, '<SOS>')
                # add_bias
                self.add_bias(label, biases, bias_seq_len)
                label_seq_len.append(len(label))
                labels.append(label)
                
                file_path = os.path.join(self.mfcc_file[1], filelist_mfcc_extend[i])
                mfcc = self.load_array_from_file(file_path)
                mfcc_seq_len.append(mfcc.shape[1])
                mfcc_features.append(mfcc)

            if len(mfcc_features)>=self.args.batch_size & len(labels)>=self.args.batch_size:
                labels = self.trans_label_to_index(labels)
                biases = self.trans_label_to_index(biases)

                # pad features and labels with constant 0
                labels = [np.pad(l, (0, max(label_seq_len)-len(l)), 'constant') for l in labels]
                mfcc_features = [np.pad(m, ((0, 0), (0, max(mfcc_seq_len)-m.shape[1])), \
                                                    'constant') for m in mfcc_features]
                biases = [np.pad(b, (0, max(bias_seq_len)-len(b)), 'constant') for b in biases]
                
                mfcc_features = np.array(mfcc_features)
                yield mfcc_features, mfcc_seq_len, labels, label_seq_len, biases, bias_seq_len

                # reset lists
                mfcc_features = []
                mfcc_seq_len = []
                labels = []
                label_seq_len = []
                biases = []
                bias_seq_len = []
                