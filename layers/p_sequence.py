import mindspore

def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    if not batch_first:
        raise NotImplementedError('batch_first=False is not supported.')
    if not enforce_sorted:
        raise NotImplementedError('enforce_sorted=False is not supported.')

    max_len = lengths
    
    print(lengths.asnumpy().shape)
    print(input.shape)
    
    if isinstance(max_len, mindspore.Tensor):
        max_len = max_len.asnumpy().item()
        
    if max_len == input.shape[1]:
        return input
    else:
        print(max_len)
        return input[:, :max_len]


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    return sequence, sequence.shape