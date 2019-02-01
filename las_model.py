import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from layers import crf_layer_likelihood, make_rnn_cell, make_attention_cell, blstm, lstm

class LAS(object):
    ''' Listen, Attend and Spell
        inspired by github 
        https://github.com/thomasschmied/Speech_Recognition_with_Tensorflow
    '''
    def __init__(self, args, vocab):
        self.args=args
        self.vocab=vocab
        self.vocab_size=len(vocab)+1
        self.eos="<EOS>"
        self.sos="<SOS>"
        self.pad='<PAD>'
      
      
    def build_model(self):
        self.init_placeholder()
        self.init_embeddings()
        self.emb_lookup()
        self.create_seq2seq()
      
    def init_placeholder(self):
        self.audios = tf.placeholder(tf.float32,
                        shape=[None, None, 494])
        self.char_ids = tf.placeholder(tf.int32,
                        shape=[None, None],
                        name='ids_target')
        self.bias_ids = tf.placeholder(tf.int32,
                        shape=[None, None],
                        name='bias_ids')
        self.audio_sequence_lengths = tf.placeholder(tf.int32,
                              shape=[None],
                              name='sequence_length_source')
        self.char_sequence_lengths = tf.placeholder(tf.int32,
                              shape=[None],
                              name='sequence_length_target')
        self.bias_sequence_lengths = tf.placeholder(tf.int32,
                              shape=[None],
                              name='sequence_length_bias')
        self.bias_attention_lengths = tf.placeholder(tf.int32,
                              shape=[None],
                              name='sequence_attention_bias')
        self.maximum_iterations = tf.reduce_max(self.char_sequence_lengths,
                            name='max_dec_len')
    
    def create_word_embedding(self, embed_name, vocab_size, embed_dim):
        """Creates embedding matrix in given shape - [vocab_size, embed_dim].
        """
        embedding = tf.get_variable(embed_name,
                      shape=[vocab_size+1, embed_dim],
                      dtype=tf.float32)
        return embedding
    
    def init_embeddings(self):
        self.embedding = self.create_word_embedding('embedding', 
                              self.vocab_size, 
                              self.args.embedding_dim)
    
    def emb_lookup(self):
        char_embedding = tf.nn.embedding_lookup(self.embedding,
                            self.char_ids,
                            name='char_embedding')
        bias_embedding = tf.nn.embedding_lookup(self.embedding,
                            self.bias_ids,
                            name='bias_embedding')
        self.char_embedding = tf.nn.dropout(char_embedding,
                          self.args.keep_probability_e,
                          name='char_embedding_dropout')
        self.bias_embedding = bias_embedding
    
    def compute_loss(self, logits):
        """Compute the loss during optimization."""
        target_output = self.char_ids
        max_time = self.maximum_iterations

        target_weights = tf.sequence_mask(self.char_sequence_lengths,
                          max_time,
                          dtype=tf.float32,
                          name='mask')
        
        if self.args.label_smoothing > 0.0:
            # Label smoothing
            label_smoothing = self.args.label_smoothing
            num_classes = array_ops.shape(logits)[2]
            target_output = tf.one_hot(target_output, num_classes)
            target_output = target_output * (1.0 - label_smoothing) \
                            + label_smoothing / tf.cast(num_classes, tf.float32)
        
        if self.args.crf_layer == True:
            log_likelihood, _ = crf_layer_likelihood(logits, 
                                                self.char_ids, 
                                                self.char_sequence_lengths)
            loss = tf.reduce_mean(-log_likelihood)
        else:
            loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                targets=target_output,
                                weights=target_weights,
                                average_across_timesteps=True,
                                average_across_batch=True, )
        
        loss *= target_weights
        loss = math_ops.reduce_sum(loss)
        total_size = math_ops.reduce_sum(target_weights)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        loss /= total_size
        
        return loss
    
    def create_seq2seq(self):
        """Creates the sequence to sequence architecture."""
        with tf.variable_scope('dynamic_seq2seq', dtype=tf.float32):
            # Radios Encoder
            encoder_outputs, encoder_state = self.build_encoder()
            # Biases Encoder
            bias_encoder_state = self.build_bias_encoder()
            
            # Decoder
            logits, sample_id, final_context_state = self.build_decoder(encoder_outputs,
                                          encoder_state,
                                          bias_encoder_state)
            self.out_logits = logits
            self.pred = sample_id
            if self.args.mode == 'TRAIN' or self.args.mode == 'FINETUNE':
                
                # Loss
                loss = self.compute_loss(logits)
                self.train_loss = loss
                self.eval_loss = loss
                self.global_step = tf.Variable(0, trainable=False)
                
                # cyclic learning rate
                if self.args.use_cyclic_lr:
                    self.args.learning_rate = self.triangular_lr(self.global_step)
                # exponential learning rate
                else:
                    self.args.learning_rate = tf.train.exponential_decay(
                      self.args.learning_rate,
                      self.global_step,
                      decay_steps=self.args.learning_rate_decay_steps,
                      decay_rate=self.args.learning_rate_decay,
                      staircase=True)
                
                # Optimizer
                opt = tf.train.AdamOptimizer(self.args.learning_rate, beta1=0.7, beta2=0.99)

                # Gradients
                if self.args.clip > 0:
                    grads, vs = zip(*opt.compute_gradients(self.train_loss))
                    grads, _ = tf.clip_by_global_norm(grads, self.args.clip)
                    self.train_op = opt.apply_gradients(zip(grads, vs),
                                      global_step=self.global_step)
                else:
                    self.train_op = opt.minimize(self.train_loss,
                                   global_step=self.global_step)

            elif self.args.mode == 'INFER':
                loss = None
                self.infer_logits, _, self.final_context_state, self.sample_id = \
                                          logits, loss, final_context_state, sample_id
                self.sample_words = self.sample_id
    
    def reshape_pyramidal(self, outputs, sequence_length):
        """
        Reshapes the given outputs, i.e. reduces the
        time resolution by 2.

        """
        # [batch_size, max_time, num_units]
        shape = tf.shape(outputs)
        batch_size, max_time = shape[0], shape[1]
        num_units = outputs.get_shape().as_list()[-1]

        pads = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
        outputs = tf.pad(outputs, pads)

        concat_outputs = tf.reshape(outputs, (batch_size, -1, num_units * 2))
        return concat_outputs, tf.floordiv(sequence_length, 2) + \
                                    tf.floormod(sequence_length, 2)

    def triangular_lr(self, current_step):
        """cyclic learning rate - exponential range."""
        step_size = self.args.learning_rate_decay_steps
        base_lr = self.args.learning_rate
        max_lr = self.args.max_lr

        cycle = tf.floor(1 + current_step / (2 * step_size))
        x = tf.abs(current_step / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * tf.maximum(0.0, tf.cast((1.0 - x), \
              dtype=tf.float32)) * (0.99999 ** tf.cast(current_step,dtype=tf.float32))
        return lr
      
    def build_encoder(self):
        with tf.variable_scope("encoder"):
            # Pyramidal bidirectional LSTM(s)
            inputs = self.audios
            seq_lengths = self.audio_sequence_lengths

            initial_state_fw = None
            initial_state_bw = None

            for n in range(self.args.num_layers_encoder):
                scope = 'pBLSTM' + str(n)
                (out_fw, out_bw), (state_fw, state_bw) = blstm(
                  self.args,
                  inputs,
                  seq_lengths,
                  self.args.rnn_size_encoder // 2,
                  scope=scope,
                  initial_state_fw=initial_state_fw,
                  initial_state_bw=initial_state_bw
                )

                inputs = tf.concat([out_fw, out_bw], -1)
                inputs, seq_lengths = self.reshape_pyramidal(inputs, seq_lengths)
                initial_state_fw = state_fw
                initial_state_bw = state_bw

            # there is a mistake ??????
            #bi_state_c = tf.concat((initial_state_fw.c, initial_state_fw.c), -1)
            #bi_state_h = tf.concat((initial_state_fw.h, initial_state_fw.h), -1)
            bi_state_c = tf.concat((initial_state_fw.c, initial_state_bw.c), -1)
            bi_state_h = tf.concat((initial_state_fw.h, initial_state_bw.h), -1)
            bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
            # there is a mistake ??????
            #encoder_state = tuple([bi_lstm_state] * self.args.num_layers_encoder)
            encoder_state = tuple([bi_lstm_state] * self.args.num_layers_decoder)

            return inputs, encoder_state
        
    def build_bias_encoder(self):
        with tf.variable_scope("bias_encoder"):
            inputs = self.bias_embedding
            seq_lengths = self.bias_sequence_lengths
            initial_state = None
            
            out, state = lstm(
              self.args,
              inputs,
              seq_lengths,
              self.args.rnn_size_encoder,
              scope='lstm_bias',
              initial_state=initial_state
            )
            
            state = tf.concat([state.c, state.h], -1)
            state = tf.expand_dims(state, 0)
            state = tf.tile(state, tf.constant([self.args.batch_size,1,1]))
            return state
    
    def build_decoder(self, encoder_outputs, encoder_state, bias_encoder_state):
        sos_id_2 = tf.cast(self.vocab[self.sos], tf.int32)
        eos_id_2 = tf.cast(self.vocab[self.eos], tf.int32)
        
        self.output_layer = Dense(self.vocab_size, name='output_projection')
        
        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self.build_decoder_cell(
              encoder_outputs,
              encoder_state,
              self.audio_sequence_lengths,
              bias_encoder_state)

            # Train
            if self.args.mode != 'INFER':

                helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                  inputs=self.char_embedding,
                  sequence_length=self.char_sequence_lengths,
                  embedding=self.embedding,
                  sampling_probability=0.5,
                  time_major=False)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                      helper,
                                      decoder_initial_state,
                                      output_layer=self.output_layer)
                
                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                  my_decoder,
                  output_time_major=False,
                  maximum_iterations=self.maximum_iterations,
                  swap_memory=False,
                  impute_finished=False,
                  scope=decoder_scope
                )
                sample_id = outputs.sample_id
                logits = outputs.rnn_output

            # Inference
            else:
                start_tokens = tf.fill([self.args.batch_size], sos_id_2)
                end_token = eos_id_2

                # Beam search
                if self.args.beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                cell=cell,
                                embedding=self.embedding,
                                start_tokens=start_tokens,
                                end_token=end_token,
                                initial_state=decoder_initial_state,
                                beam_width=self.args.beam_width,
                                output_layer=self.output_layer,
                  )

                # Greedy
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding,
                                              start_tokens,
                                              end_token)

                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                           helper,
                                           decoder_initial_state,
                                           output_layer=self.output_layer)
                if self.args.inference_targets:
                    maximum_iterations = self.maximum_iterations
                else:
                    maximum_iterations = None

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                                              my_decoder,
                                              maximum_iterations=maximum_iterations,
                                              output_time_major=False,
                                              impute_finished=False,
                                              swap_memory=False,
                                              scope=decoder_scope)

                if self.args.beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = tf.no_op()
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state
    
    def build_decoder_cell(self, encoder_outputs, encoder_state,
                   audio_sequence_lengths, bias_encoder_state):
        """Builds the attention decoder cell. If mode is inference performs tiling
          Passes last encoder state.
        """

        memory = encoder_outputs
        memory_bias = bias_encoder_state

        if self.args.mode == 'INFER' and self.args.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(memory,
                                multiplier=self.args.beam_width)
            memory_bias = tf.contrib.seq2seq.tile_batch(memory_bias,
                                multiplier=self.args.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,
                                    multiplier=self.args.beam_width)
            audio_sequence_lengths = tf.contrib.seq2seq.tile_batch(audio_sequence_lengths,
                                        multiplier=self.args.beam_width)
            batch_size = self.args.batch_size * self.args.beam_width
            audios_shape = tf.shape(self.audios)[0] * self.args.beam_width
        else:
            batch_size = self.args.batch_size
            audios_shape = tf.shape(self.audios)[0]

        if self.args.num_layers_decoder is not None:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
              [make_rnn_cell(self.args.rnn_size_decoder, self.args) for _ in
               range(self.args.num_layers_decoder)])
        else:
            lstm_cell = make_rnn_cell(self.args.rnn_size_decoder, self.args)

        # attention cell
        cell = make_attention_cell(lstm_cell,
                        self.args.rnn_size_decoder,
                        memory,
                        memory_bias,
                        audio_sequence_lengths,
                        self.args.attention_type,
                        self.args.attention_type_bias,
                        self.bias_attention_lengths,
                        self.args)
        
        decoder_initial_state = cell.zero_state(audios_shape, tf.float32).\
                                          clone(cell_state=encoder_state)
        
        return cell, decoder_initial_state
      