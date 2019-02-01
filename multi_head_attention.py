import collections
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism

_zero_state_tensors = rnn_cell_impl._zero_state_tensors

def _scaled_dot(qs, ks, size_per_head):
    '''using the querys and keys to compute scaled dot value
       here we do not have to do mask, because it will be masked in 
       function _prepare_memory in attention_wrapper.py
    '''
    o1 = tf.matmul(qs, ks, transpose_b=True)
    o2 = o1 / (size_per_head**0.5)

    return o2

def _scaled_dot_attention(query, keys, normalize, size_per_head):
    '''invoke the scale dot function
       size_per_head: the dims in every head
       
       TODO: the normalize
    '''
    outputs = tf.reduce_sum(_scaled_dot(query, keys, size_per_head), [2])

    return outputs

def split_heads(q, k, num_heads, size_per_head):

    def split_last_dimension_then_transpose(tensor, num_heads, size_per_head):
        t_shape = tf.shape(tensor)
        tensor = tf.reshape(tensor, [t_shape[0], -1] + [num_heads, size_per_head])
        return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

    qs = split_last_dimension_then_transpose(q, num_heads, size_per_head)
    ks = split_last_dimension_then_transpose(k, num_heads, size_per_head)
    return qs, ks


class MultiHeadAttention(_BaseAttentionMechanism):
    '''rewrite the class attention_mechanism for our personal requirement
    '''
    def __init__(self,
                num_units,
                memory,
                memory_sequence_length=None,
                normalize=False,
                probability_fn=None,
                score_mask_value=float("-inf"),
                name="MultiHeadAttention"):
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(MultiHeadAttention, self).__init__(
          query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False),
          memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False),
          memory=memory,
          probability_fn=wrapped_probability_fn,
          memory_sequence_length=memory_sequence_length,
          score_mask_value=score_mask_value,
          name=name)
        self.memory = memory
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, score, previous_alignments):
        alignments = self._probability_fn(score, previous_alignments)
        return alignments


class MyAttentionMechanism():
    '''this class is used to parallel multiple heads
    '''
    def __init__(self,
                  num_heads,
                  size_per_head,
                  memory,
                  memory_sequence_length=None,
                  normalize=False,
                  probability_fn=None,
                  score_mask_value=float("-inf"),
                  name="MyAttentionMechanism"):
        self.query_layer=layers_core.Dense(
          num_heads*size_per_head, name="query_layer", use_bias=False)
        self.memory_layer=layers_core.Dense(
          num_heads*size_per_head, name="memory_layer", use_bias=False)
        self.memory=memory
        self._normalize=normalize
        self._num_heads=num_heads
        self._size_per_head=size_per_head

    def __call__(self, query):
        with variable_scope.variable_scope(None, "my_attention_mechanism", [query]):
            q = self.query_layer(query)
            k = self.memory_layer(self.memory)
            q = array_ops.expand_dims(q, 1)
            qs, ks = split_heads(q, k, self._num_heads, self._size_per_head)
            score = _scaled_dot_attention(qs, ks, self._normalize, self._size_per_head)
        return score


class AttentionWrapperState(
      collections.namedtuple("AttentionWrapperState",
                             ("cell_state", "bias_state", "attention", "attention_bias",
                              "time", "alignments", "alignments_bias",
                              "alignment_history", "alignment_history_bias"))):
    """`namedtuple` storing the state of a `AttentionWrapper`.
    Contains:
      - `cell_state`: The state of the wrapped `RNNCell` at the previous time
        step.
      - `attention`: The attention emitted at the previous time step.
      - `time`: int32 scalar containing the current time step.
      - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
         emitted at the previous time step for each attention mechanism.
      - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
         containing alignment matrices from all time steps for each attention
         mechanism. Call `stack()` on each to convert to a `Tensor`.
    """

    def clone(self, **kwargs):
        """Clone this object, overriding components provided by kwargs.
        Example:
        ```python
        initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
        initial_state = initial_state.clone(cell_state=encoder_state)
        ```
        Args:
          **kwargs: Any properties of the state object to replace in the returned
            `AttentionWrapperState`.
        Returns:
          A new `AttentionWrapperState` whose properties are the same as
          this one, except any overridden properties as provided in `kwargs`.
        """
        return super(AttentionWrapperState, self)._replace(**kwargs)

class AttentionWrapper(rnn_cell_impl.RNNCell):
    """rewrite the AttentionWrapper class for invoking our own mechanism class
        reference the tensorflow source code about this class detail 
    """

    def __init__(self,
                cell,
                attention_mechanism,
                attention_mechanism_for_bias,
                my_attention_mechanism,
                my_attention_mechanism_bias,
                attention_layer_size=None,
                alignment_history=False,
                cell_input_fn=None,
                output_attention=True,
                initial_cell_state=None,
                name=None):
        """Construct the `AttentionWrapper`.
        **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
        `AttentionWrapper`, then you must ensure that:
        - The encoder output has been tiled to `beam_width` via
          @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
        - The `batch_size` argument passed to the `zero_state` method of this
          wrapper is equal to `true_batch_size * beam_width`.
        - The initial state created with `zero_state` above contains a
          `cell_state` value containing properly tiled final state from the
          encoder.
        An example:
        ```
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
            encoder_final_state, multiplier=beam_width)
        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
            sequence_length, multiplier=beam_width)
        attention_mechanism = MyFavoriteAttentionMechanism(
            num_units=attention_depth,
            memory=tiled_inputs,
            memory_sequence_length=tiled_sequence_length)
        attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
        decoder_initial_state = attention_cell.zero_state(
            dtype, batch_size=true_batch_size * beam_width)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=tiled_encoder_final_state)
        ```
        Args:
          cell: An instance of `RNNCell`.
          attention_mechanism: A list of `AttentionMechanism` instances or a single
            instance.
          attention_layer_size: A list of Python integers or a single Python
            integer, the depth of the attention (output) layer(s). If None
            (default), use the context as attention at each time step. Otherwise,
            feed the context and cell output into the attention layer to generate
            attention at each time step. If attention_mechanism is a list,
            attention_layer_size must be a list of the same length.
          alignment_history: Python boolean, whether to store alignment history
            from all time steps in the final output state (currently stored as a
            time major `TensorArray` on which you must call `stack()`).
          cell_input_fn: (optional) A `callable`.  The default is:
            `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
          output_attention: Python bool.  If `True` (default), the output at each
            time step is the attention value.  This is the behavior of Luong-style
            attention mechanisms.  If `False`, the output at each time step is
            the output of `cell`.  This is the beahvior of Bhadanau-style
            attention mechanisms.  In both cases, the `attention` tensor is
            propagated to the next time step via the state and is used there.
            This flag only controls whether the attention mechanism is propagated
            up to the next cell in an RNN stack or to the top RNN output.
          initial_cell_state: The initial state value to use for the cell when
            the user calls `zero_state()`.  Note that if this value is provided
            now, and the user uses a `batch_size` argument of `zero_state` which
            does not match the batch size of `initial_cell_state`, proper
            behavior is not guaranteed.
          name: Name to use when creating ops.
        Raises:
          TypeError: `attention_layer_size` is not None and (`attention_mechanism`
            is a list but `attention_layer_size` is not; or vice versa).
          ValueError: if `attention_layer_size` is not None, `attention_mechanism`
            is a list, and its length does not match that of `attention_layer_size`.
        """
        super(AttentionWrapper, self).__init__(name=name)
        if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
            raise TypeError(
                "cell must be an RNNCell, saw type: %s" % type(cell).__name__)

        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = attention_mechanism
            for attention_mechanism in attention_mechanisms:
                if not isinstance(attention_mechanism, AttentionMechanism):
                    raise TypeError(
                        "attention_mechanism must contain only instances of "
                        "AttentionMechanism, saw type: %s"
                        % type(attention_mechanism).__name__)
        else:
            self._is_multi = False
            if not isinstance(attention_mechanism, AttentionMechanism):
                raise TypeError(
                    "attention_mechanism must be an AttentionMechanism or list of "
                    "multiple AttentionMechanism instances, saw type: %s"
                    % type(attention_mechanism).__name__)
            attention_mechanisms = (attention_mechanism,)
        
        # this part is for bias 
        if isinstance(attention_mechanism_for_bias, (list, tuple)):
            self._is_multi_bias = True
            attention_mechanisms_for_bias = attention_mechanism_for_bias
            for attention_mechanism_for_bias in attention_mechanisms_for_bias:
                if not isinstance(attention_mechanism_for_bias, AttentionMechanism):
                    raise TypeError(
                        "attention_mechanism_for_bias must contain only instances of "
                        "AttentionMechanism, saw type: %s"
                        % type(attention_mechanism_for_bias).__name__)
        else:
            self._is_multi_bias = False
            if not isinstance(attention_mechanism_for_bias, AttentionMechanism):
                raise TypeError(
                    "attention_mechanism_for_bias must be an AttentionMechanism or list of "
                    "multiple AttentionMechanism instances, saw type: %s"
                    % type(attention_mechanism_for_bias).__name__)
            attention_mechanisms_for_bias = (attention_mechanism_for_bias,)
        
        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, attention: array_ops.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                    "cell_input_fn must be callable, saw type: %s"
                    % type(cell_input_fn).__name__)

        if attention_layer_size is not None:
            attention_layer_sizes = tuple(
                attention_layer_size
                if isinstance(attention_layer_size, (list, tuple))
                else (attention_layer_size,))
            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError(
                    "If provided, attention_layer_size must contain exactly one "
                    "integer per attention_mechanism, saw: %d vs %d"
                    % (len(attention_layer_sizes), len(attention_mechanisms)))
            self._attention_layers = tuple(
                layers_core.Dense(
                    attention_layer_size, name="attention_layer", use_bias=False)
                for attention_layer_size in attention_layer_sizes)
            self._attention_layer_size = sum(attention_layer_sizes)
        else:
            self._attention_layers = None
            self._attention_layer_size = sum(
                attention_mechanism.values.get_shape()[-1].value
                for attention_mechanism in attention_mechanisms)
            # this part is for bias
            self._attention_layer_size_bias = sum(
                attention_mechanism_bias.values.get_shape()[-1].value
                for attention_mechanism_bias in attention_mechanisms_for_bias)


        self._cell = cell
        # my own attention mechanism
        self._attention_mechanisms = attention_mechanisms
        # attention mechanism for bias
        self._attention_mechanisms_bias = attention_mechanisms_for_bias
        self._my_attention_mechanism = my_attention_mechanism
        self._my_attention_mechanism_bias = my_attention_mechanism_bias
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        with ops.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or array_ops.shape(final_state_tensor)[0])
                error_message = (
                    "When constructing AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.  Are you using "
                    "the BeamSearchDecoder?  You may need to tile your initial state "
                    "via the tf.contrib.seq2seq.tile_batch function with argument "
                    "multiple=beam_width.")
                with ops.control_dependencies(
                    self._batch_size_checks(state_batch_size, error_message)):
                  self._initial_cell_state = nest.map_structure(
                      lambda s: array_ops.identity(s, name="check_initial_cell_state"),
                      initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [check_ops.assert_equal(batch_size,
                                       attention_mechanism.batch_size,
                                       message=error_message)
                for attention_mechanism in self._attention_mechanisms]

    def _item_or_tuple(self, seq, bias):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.
        Args:
          seq: A non-empty sequence of items or generator.
        Returns:
           Either the values in the sequence as a tuple if AttentionMechanism(s)
           were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if bias:
            if self._is_multi_bias:
                return t
            else:
                return t[0]
        else:
            if self._is_multi:
                return t
            else:
                return t[0]

    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.
        Returns:
          An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return AttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._item_or_tuple(
                (a.alignments_size for a in self._attention_mechanisms), False),
            alignment_history=self._item_or_tuple(
                (() for _ in self._attention_mechanisms), False),
            bias_state=self._cell.state_size,
            attention_bias=self._attention_layer_size_bias,
            alignments_bias=self._item_or_tuple(
                (a.alignments_size for a in self._attention_mechanisms_bias), True),
            alignment_history_bias=self._item_or_tuple(
                (() for _ in self._attention_mechanisms_bias), True))  # sometimes a TensorArray

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.
        Returns:
          An `AttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.
        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output has "
                "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                "the batch_size= argument passed to zero_state is "
                "batch_size * beam_width.")
            with ops.control_dependencies(
                  self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return AttentionWrapperState(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=dtypes.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                              dtype),
                alignments=self._item_or_tuple(
                    (attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms), False),
                alignment_history=self._item_or_tuple(
                    (tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                                 dynamic_size=True)
                    if self._alignment_history else ()
                    for _ in self._attention_mechanisms), False),
                bias_state=cell_state,
                attention_bias=_zero_state_tensors(self._attention_layer_size_bias, batch_size,
                                              dtype),
                alignments_bias=self._item_or_tuple(
                    (attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms_bias), True),
                alignment_history_bias=self._item_or_tuple(
                    (tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                                 dynamic_size=True)
                    if self._alignment_history else ()
                    for _ in self._attention_mechanisms_bias), True))

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead."  % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        # add the same calculation on bias
        bias_inputs = self._cell_input_fn(inputs, state.attention_bias)
        bias_state = state.bias_state
        bias_output, next_bias_state = self._cell(bias_inputs, bias_state)
        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)):
          cell_output = array_ops.identity(
              cell_output, name="checked_cell_output")
          # add the same calculation on bias
          bias_output = array_ops.identity(
              bias_output, name="checked_bias_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []
        # calculate the multi_heads score ***
        score = self._my_attention_mechanism(cell_output)
        score = tf.transpose(score, [1, 0, 2])
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments = _compute_attention(
                attention_mechanism, score[i], previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None)  #***
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)
        
        # add the same calculation on bias
        if self._is_multi_bias:
            previous_alignments_bias = state.alignments_bias
            previous_alignment_history_bias = state.alignment_history_bias
        else:
            previous_alignments_bias = [state.alignments_bias]
            previous_alignment_history_bias = [state.alignment_history_bias]
        
        all_alignments_bias = []
        all_attentions_bias = []
        all_histories_bias = []
        if self._is_multi_bias:
            score = self._my_attention_mechanism_bias(bias_output)
            score = tf.transpose(score, [1, 0, 2])
            temp_output = score
        else:
            temp_output = bias_output
        for i, attention_mechanism in enumerate(self._attention_mechanisms_bias):
            attention, alignments = _compute_attention(
                attention_mechanism, temp_output[i] if self._is_multi_bias else temp_output, 
                previous_alignments_bias[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history_bias[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments_bias.append(alignments)
            all_histories_bias.append(alignment_history)
            all_attentions_bias.append(attention)
        
        attention = array_ops.concat(all_attentions, 1)
        attention_bias = array_ops.concat(all_attentions_bias, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            bias_state=next_bias_state,
            attention=attention,
            attention_bias=attention_bias,
            alignments=self._item_or_tuple(all_alignments, False),
            alignments_bias=self._item_or_tuple(all_alignments_bias, True),
            alignment_history=self._item_or_tuple(all_histories, False),
            alignment_history_bias=self._item_or_tuple(all_histories_bias, True))
        
        if self._output_attention:
            return attention_bias, next_state
        else:
            return tf.concat([cell_output,bias_output],-1), next_state