from typing import Optional, List

import torch
from torch import Tensor


class BufferManager:
    def __init__(self, buffer_size: int = 128):
        self.buffer_size = buffer_size
        self.key_buffer: Optional[Tensor] = None
        self.value_buffer: Optional[Tensor] = None
        self.overflow_keys: List[Tensor] = []
        self.overflow_values: List[Tensor] = []

    def add(self, key: Tensor, value: Tensor):
        if self.key_buffer is None:
            self.key_buffer = key
            self.value_buffer = value
        else:
            self.key_buffer = torch.cat([self.key_buffer, key], dim=-2)
            self.value_buffer = torch.cat([self.value_buffer, value], dim=-2)

    def has_overflow(self) -> bool:
        if self.key_buffer is None:
            return False
        return self.key_buffer.shape[-2] > self.buffer_size

    def flush_overflow(self):
        if self.key_buffer is None or not self.has_overflow():
            return None, None

        overflow_len = self.key_buffer.shape[-2] - self.buffer_size
        overflow_keys = self.key_buffer[..., :overflow_len, :]
        overflow_values = self.value_buffer[..., :overflow_len, :]

        self.key_buffer = self.key_buffer[..., overflow_len:, :]
        self.value_buffer = self.value_buffer[..., overflow_len:, :]

        return overflow_keys, overflow_values

    def get_buffer(self):
        return self.key_buffer, self.value_buffer

    def get_seq_len(self) -> int:
        if self.key_buffer is None:
            return 0
        return self.key_buffer.shape[-2]

    def clear(self):
        self.key_buffer = None
        self.value_buffer = None
        self.overflow_keys.clear()
        self.overflow_values.clear()
