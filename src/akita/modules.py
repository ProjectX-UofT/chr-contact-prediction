import torch
import torch.nn as nn
import numpy as np

class StochasticReverseComplement(nn.Module):
  """Stochastically reverse complement a one hot encoded DNA sequence."""
  def __init__(self):
    super(StochasticReverseComplement, self).__init__()
  def forward(self, seq_1hot, training=None):
    if training:
      rc_seq_1hot = torch.flip(seq_1hot, dims=[-1])
      rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[1])
      reverse_bool = torch.randn(1) > 0
      src_seq_1hot = torch.where(reverse_bool, rc_seq_1hot, seq_1hot)
      return src_seq_1hot, reverse_bool
    else:
      return seq_1hot, torch.Tensor(False)


class StochasticShift(nn.Module):
  """Stochastically shift a one hot encoded DNA sequence."""
  def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
    super(StochasticShift, self).__init__()
    self.shift_max = shift_max
    self.symmetric = symmetric
    if self.symmetric:
      self.augment_shifts = torch.range(-self.shift_max, self.shift_max+1)
    else:
      self.augment_shifts = torch.range(0, self.shift_max+1)
    self.pad = pad

  def forward(self, seq_1hot, training=None):
    if training:
      shift_i = torch.randint(0, len(self.augment_shifts),
                              (), dtype=torch.int64)
      shift = self.augment_shifts[shift_i]
      sseq_1hot = torch.where((shift != 0), shift_sequence(seq_1hot, shift), seq_1hot)
      return sseq_1hot
    else:
      return seq_1hot

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'shift_max': self.shift_max,
      'symmetric': self.symmetric,
      'pad': self.pad
    })
    return config

def shift_sequence(seq, shift, pad_value=0.25):
  """Shift a sequence left or right by shift_amount.

  Args:
  seq: [batch_size, seq_length, seq_depth] sequence
  shift: signed shift value (tf.int32 or int)
  pad_value: value to fill the padding (primitive or scalar tf.Tensor)
  """
  if len(seq.shape) != 3:
      raise ValueError('input sequence should be rank 3')
  input_shape = seq.shape
  pad = pad_value * torch.ones_like(seq[:, 0:torch.abs(shift), :])

  def _shift_right(_seq):
    # shift is positive
    sliced_seq = _seq[:, :-shift:, :]
    return torch.concat((pad, sliced_seq), dim=1)

  def _shift_left(_seq):
    # shift is negative
    sliced_seq = _seq[:, -shift:, :]
    return torch.concat((pad, sliced_seq), dim=1)

  sseq = torch.where(shift[0], _shift_right(seq), _shift_left(seq))
  sseq.set_shape(input_shape)

  return sseq

if __name__ == "__main__":
    t = torch.Tensor(np.arange(20).reshape((1, 5, 4)))
    src = StochasticShift()
    print()