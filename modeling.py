
import paddle.fluid as fluid
import math
import copy
import numpy as np




class PaddleBert(object):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = input_ids.shape
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = fluid.layers.ones(shape=[batch_size, seq_length], dtype='int32')
        if token_type_ids is None:
            token_type_ids = fluid.layers.zeros(shape=[batch_size, seq_length], dtype='int32')

        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            use_one_hot_embeddings=True)

        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

def gelu(x):
    cdf = 0.5 * (1.0 + fluid.layers.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * fluid.layers.pow(x, 3)))))
    return x * cdf




def layer_norm_and_dropout(input_tensor, dropout_prob):
    output_tensor = fluid.layers.layer_norm(input_tensor)
    output_tensor = fluid.layers.dropout(output_tensor, dropout_prob)
    return output_tensor


def embedding_lookup(
        input_ids,
        vocab_size,
        embedding_size=256,
        initializer_range=0.02,
        use_onehot_embeddings=True
):
    if len(input_ids.shape) == 2:
        input_ids = fluid.layers.nn.reshape(input_ids, [input_ids.shape[0], input_ids.shape[1], -1])
    embedding_table = fluid.layers.create_parameter(shape=[vocab_size, embedding_size], dtype='float32',
                                                    default_initializer=fluid.initializer.NormalInitializer())
    flat_input_ids = fluid.layers.nn.reshape(input_ids, [-1])

    if use_onehot_embeddings:
        one_hot_input_ids = fluid.layers.one_hot(input_ids,
                                                 depth=vocab_size)  # paddlepaddle one_hot layer can only accept tensor whose shape is [?,1]
        output = fluid.layers.matmul(one_hot_input_ids, embedding_table)
    else:
        output = fluid.layers.gather(embedding_table, flat_input_ids)

    input_shape = input_ids.shape

    output = fluid.layers.nn.reshape(output, list(input_shape[0:-1]) + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def embedding_postprocessor(
        input_tensor,
        use_token_type=False,
        token_type_ids=None,
        token_type_vocab_size=16,
        token_type_embedding_name='token_type_embedding',
        use_position_embeddings=True,
        position_embedding_name='position_embeddings',
        initializer_range=0.02,
        max_position_embeddings=512,
        droupout_prob=0.1
):
    input_shape = input_tensor.shape
    batch_szie = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        if len(token_type_ids.shape) == 2:
            token_type_ids = fluid.layers.nn.reshape(token_type_ids,
                                                     [token_type_ids.shape[0], token_type_ids.shape[1], -1])
        token_type_table = fluid.layers.create_parameter(shape=[token_type_vocab_size, width], dtype='float32',
                                                         default_initializer=fluid.initializer.NormalInitializer(),
                                                         name=token_type_embedding_name)

        one_hot_token_type_ids = fluid.layers.one_hot(token_type_ids,
                                                      depth=token_type_vocab_size)  # paddlepaddle one_hot layer can only accept tensor whose shape is [?,1]
        token_type_embeddings = fluid.layers.matmul(one_hot_token_type_ids, token_type_table)
        output += token_type_embeddings

    if use_position_embeddings:
        full_position_embeddings = fluid.layers.create_parameter(shape=[max_position_embeddings, width],
                                                                 dtype='float32',
                                                                 default_initializer=fluid.initializer.NormalInitializer(),
                                                                 name=position_embedding_name)
        positional_embeddings = fluid.layers.slice(full_position_embeddings, starts=[0, 0], ends=[seq_length, -1])
        num_dims = len(positional_embeddings.shape)
        postion_broadcast_shape = []
        for _ in range(num_dims - 2):
            postion_broadcast_shape.append(1)
        postion_broadcast_shape.extend([seq_length, width])
        positional_embeddings = fluid.layers.reshape(positional_embeddings, postion_broadcast_shape)

        output += positional_embeddings
    output = layer_norm_and_dropout(output, droupout_prob)
    return output


def transpose_to_2DTensor(input_tensor):
    width = input_tensor.shape[-1]
    output_tensor = fluid.layers.nn.reshape(input_tensor, [-1, width])
    return output_tensor


def transpose_for_attention_scores(input_tensor, batch_size, num_attention_head, seq_length, width):
    output_tensor = fluid.layers.nn.reshape(input_tensor, [batch_size, seq_length, num_attention_head, width])
    output_tensor = fluid.layers.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor


def attention_layer(
        from_tensor,
        to_tensor,
        attention_mask=None,
        num_attention_head=1,
        per_head_size=512,
        attention_drop_out_prob=0.1,
        query_act=None,
        key_act=None,
        value_act=None,
        batch_size=None,
        from_seq_length=None,
        to_seq_length=None
):
    extened_from_tensor = transpose_to_2DTensor(from_tensor)
    extened_to_tensor = transpose_to_2DTensor(to_tensor)

    query_layer = fluid.layers.fc(extened_from_tensor, size=num_attention_head * per_head_size, act=query_act,
                                  name='query_layer')
    key_layer = fluid.layers.fc(extened_to_tensor, size=num_attention_head * per_head_size, act=key_act,
                                name='key_layer')
    value_layer = fluid.layers.fc(extened_to_tensor, size=num_attention_head * per_head_size, act=value_act,
                                  name='value_layer')

    query_layer = transpose_for_attention_scores(query_layer, batch_size, num_attention_head, from_seq_length,
                                                 per_head_size)

    key_layer = transpose_for_attention_scores(key_layer, batch_size, num_attention_head, to_seq_length, per_head_size)
    attention_scores = fluid.layers.matmul(query_layer, key_layer, transpose_y=True)
    attention_scores = fluid.layers.scale(attention_scores, scale=1.0 / math.sqrt(float(per_head_size)))

    attention_probs = fluid.layers.softmax(attention_scores)

    attention_probs = fluid.layers.dropout(attention_probs, attention_drop_out_prob)

    value_layer = fluid.layers.reshape(value_layer, [batch_size, to_seq_length, num_attention_head, per_head_size])
    value_layer = fluid.layers.transpose(value_layer, [0, 2, 1, 3])

    context_layer = fluid.layers.matmul(attention_probs, value_layer)
    context_layer = fluid.layers.transpose(context_layer, [0, 2, 1, 3])

    context_layer = fluid.layers.reshape(context_layer,
                                         [batch_size, from_seq_length, num_attention_head * per_head_size])

    return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = input_tensor.shape
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    if width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (width, hidden_size))

    prev_output = transpose_to_2DTensor(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        layer_input = prev_output
        attention_heads = []
        attention_head = attention_layer(
            from_tensor=layer_input,
            to_tensor=layer_input,
            attention_mask=attention_mask,
            num_attention_head=num_attention_heads,
            per_head_size=attention_head_size,
            attention_drop_out_prob=attention_probs_dropout_prob,
            batch_size=batch_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length
        )
        attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            attention_output = fluid.layers.concat(attention_heads, axis=-1)

        attention_output = fluid.layers.fc(
            attention_output,
            hidden_size,
            param_attr=fluid.initializer.NormalInitializer()
        )

        attention_output = fluid.layers.dropout(attention_output, hidden_dropout_prob)
        attention_output = fluid.layers.layer_norm(attention_output)

        intermediate_output = fluid.layers.fc(
            attention_output,
            intermediate_size,
            param_attr=fluid.initializer.NormalInitializer()
        )

        layer_output = fluid.layers.fc(
            intermediate_output,
            hidden_size,
            param_attr=fluid.initializer.NormalInitializer()
        )

        layer_output = fluid.layers.dropout(layer_output, hidden_dropout_prob)
        layer_output = fluid.layers.layer_norm(layer_output)
        prev_output = layer_input
        all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = fluid.layers.reshape(layer_output, [batch_size, seq_length, hidden_size])
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = fluid.layers.reshape(prev_output, [batch_size, seq_length, hidden_size])
        return final_output
