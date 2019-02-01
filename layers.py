import tensorflow as tf

from multi_head_attention import MultiHeadAttention
from multi_head_attention import MyAttentionMechanism
from multi_head_attention import AttentionWrapper




def crf_layer_likelihood(logits, targets, seq_len):
    '''add a crf layer
    '''
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                                            logits, 
                                            targets, 
                                            seq_len)
    return log_likelihood, transition_params


def make_rnn_cell(rnn_size, args):
    """Creates LSTM cell wrapped with dropout.
    """
    cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                       input_keep_prob=args.keep_probability_i,
                       output_keep_prob=args.keep_probability_o,
                       state_keep_prob=args.keep_probability_h)
    return cell
    
def make_attention_cell(dec_cell, 
                        rnn_size, 
                        enc_output, 
                        bias_output, 
                        lengths, 
                        att_type, 
                        att_type_bias, 
                        bias_lengths, 
                        args):
    """Wraps the given cell with Attention.
    Args:
      dec_cell: the RNNCell for decoder.
      rnn_size: Integer. Number of hidden units to use for
            rnn cell.
      inputs: Array of input points.
      enc_output: encoder outputs in erery step.
      bias_output: bias representations.
      lengths: Array of integers. Sequence lengths of the
            input points.
      att_type: attention type for encoder.
      att_type_bias: attention for bias.
      bias_lengths: number of the bias words.

    Returns: a new Cell wrapped with attention.

    """
    if att_type=='BahdanauAttention':
    # if the attention type is BahdanauAttention, the bias has not been implemented
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                      num_units=rnn_size,
                                      memory=enc_output,
                                      memory_sequence_length=lengths,
                                      name='BahdanauAttention')

        return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                              attention_mechanism=attention_mechanism,
                              attention_layer_size=None,
                              output_attention=False)
    
    elif att_type=='MultiHeadAttention' and att_type_bias=='MultiHeadAttention':
    # multi_head_attention implement
        size_per_head = int(rnn_size/args.num_heads)
        my_attention_mechanism = MyAttentionMechanism(num_heads=args.num_heads,
                              size_per_head=size_per_head,
                              memory=enc_output,
                              memory_sequence_length=lengths,
                              name='MultiHeadAttention')
        my_attention_mechanism_bias = MyAttentionMechanism(num_heads=args.num_heads,
                              size_per_head=size_per_head,
                              memory=bias_output,
                              memory_sequence_length=bias_lengths,
                              name='MultiHeadAttentionBias')
        
        attention_mechanisms = []
        attention_mechanisms_for_bias = []
        for i in range(args.num_heads):
            attention_mechanism = MultiHeadAttention(num_units=rnn_size,
                                          memory=enc_output,
                                          memory_sequence_length=lengths,
                                          name='MultiHeadAttention')
            attention_mechanism_for_bias = MultiHeadAttention(num_units=rnn_size,
                                          memory=bias_output,
                                          memory_sequence_length=bias_lengths,
                                          name='MultiHeadAttentionBias')
            attention_mechanisms.append(attention_mechanism)
            attention_mechanisms_for_bias.append(attention_mechanism_for_bias)
        
        return AttentionWrapper(cell=dec_cell,
                    attention_mechanism=attention_mechanisms,
                    attention_mechanism_for_bias=attention_mechanisms_for_bias,
                    my_attention_mechanism=my_attention_mechanism,
                    my_attention_mechanism_bias=my_attention_mechanism_bias,
                    attention_layer_size=None,
                    output_attention=False)
    
    elif att_type=='MultiHeadAttention' and att_type_bias=='BahdanauAttention':
        if args.num_heads>1:
            raise ValueError("it's illegal if the num_heads>1 and att_type_bias \
                              equals BahdanauAttention at the same time, please set \
                              num_heads 1 or set att_type_bias MultiHeadAttention")
        size_per_head = int(rnn_size/args.num_heads)
        my_attention_mechanism = MyAttentionMechanism(num_heads=args.num_heads,
                                                size_per_head=size_per_head,
                                                memory=enc_output,
                                                memory_sequence_length=lengths,
                                                name='MultiHeadAttention')
        
        attention_mechanisms_for_bias = tf.contrib.seq2seq.BahdanauAttention(
                                        num_units=rnn_size,
                                        memory=bias_output,
                                        memory_sequence_length=bias_lengths,
                                        name='BahdanauAttention')
        attention_mechanisms = []
        for i in range(args.num_heads):
            attention_mechanism = MultiHeadAttention(num_units=rnn_size,
                                                    memory=enc_output,
                                                    memory_sequence_length=lengths,
                                                    name='MultiHeadAttention')
            attention_mechanisms.append(attention_mechanism)
        
        return AttentionWrapper(cell=dec_cell,
                    attention_mechanism=attention_mechanisms,
                    attention_mechanism_for_bias=attention_mechanisms_for_bias,
                    my_attention_mechanism=my_attention_mechanism,
                    my_attention_mechanism_bias=None,
                    attention_layer_size=None,
                    output_attention=False)
                    
def blstm(args,
          inputs,
          seq_length,
          rnn_size,
          scope=None,
          initial_state_fw=None,
          initial_state_bw=None
          ):
    """
    Creates a bidirectional lstm.
    Args:
      inputs: Array of input points.
      seq_length: Array of integers. Sequence lengths of the
            input points.
      rnn_size: Integer. Number of hidden units to use for
            rnn cell.
      scope: String.
      initial_state_fw: Initial state of foward cell.
      initial_state_bw: Initial state of backward cell.

    Returns: Tuple of fw and bw output.
         Tuple of fw and bw state.

    """
    fw_cell = make_rnn_cell(rnn_size, args)
    bw_cell = make_rnn_cell(rnn_size, args)

    (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=fw_cell,
      cell_bw=bw_cell,
      inputs=inputs,
      sequence_length=seq_length,
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      dtype=tf.float32,
      scope=scope
    )

    return (out_fw, out_bw), (state_fw, state_bw)
    
def lstm(args,
        inputs,
        seq_length,
        rnn_size,
        scope=None,
        initial_state=None
        ):
    """
    Creates a bidirectional lstm.
    Args:
      inputs: Array of input points.
      seq_length: Array of integers. Sequence lengths of the
            input points.
      rnn_size: Integer. Number of hidden units to use for
            rnn cell.
      scope: String.
      initial_state_fw: Initial state.

    Returns: output and state.
    """
    cell = make_rnn_cell(rnn_size, args)

    out, state = tf.nn.dynamic_rnn(
      cell=cell,
      inputs=inputs,
      sequence_length=seq_length,
      initial_state=initial_state,
      dtype=tf.float32,
      scope=scope
    )

    return out, state