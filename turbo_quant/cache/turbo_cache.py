from typing import Optional, List, Tuple, Dict

import torch
from torch import Tensor

from turbo_quant.config import TurboQuantConfig
from turbo_quant.cache.outlier_detector import OutlierDetector
from turbo_quant.cache.buffer_manager import BufferManager
from turbo_quant.quantizers.turbo_mse import TurboQuantMSE
from turbo_quant.quantizers.turbo_prod import TurboQuantProd, QuantizedKeyProd
from turbo_quant.quantizers.qjl import QJLQuantizer
from turbo_quant.quantizers.value_quant import ValueQuantizer


class LayerCache:
    def __init__(self):
        self.quantized_key_indices: Optional[Tensor] = None
        self.quantized_key_norms: Optional[Tensor] = None
        self.quantized_key_qjl_bits: Optional[Tensor] = None
        self.quantized_key_residual_norms: Optional[Tensor] = None

        self.outlier_key_indices: Optional[Tensor] = None
        self.outlier_key_norms: Optional[Tensor] = None
        self.outlier_key_qjl_bits: Optional[Tensor] = None
        self.outlier_key_residual_norms: Optional[Tensor] = None

        self.quantized_values: Optional[Tensor] = None
        self.value_scales: Optional[Tensor] = None
        self.value_zero_points: Optional[Tensor] = None

        self.outlier_channel_indices: Optional[Tensor] = None
        self.quantized_seq_len: int = 0


class TurboQuantCache:
    def __init__(self, config: TurboQuantConfig, num_layers: int, num_heads: int, head_dim: int):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.layers: List[LayerCache] = [LayerCache() for _ in range(num_layers)]
        self.buffers: List[BufferManager] = [BufferManager(config.buffer_size) for _ in range(num_layers)]
        self.outlier_detector = OutlierDetector(config.num_outlier_channels, config.outlier_threshold_percentile)
        self.value_quantizer = ValueQuantizer(config.value_bits, config.device, config.dtype)

        regular_dim = head_dim - config.num_outlier_channels
        self.regular_quantizers: Dict[int, object] = {}
        self.outlier_quantizers: Dict[int, object] = {}

        for layer_idx in range(num_layers):
            self.regular_quantizers[layer_idx] = self._create_quantizer(regular_dim, config.key_bits, config.device, config.dtype)
            self.outlier_quantizers[layer_idx] = self._create_quantizer(config.num_outlier_channels, config.outlier_bits, config.device, config.dtype)

        self._initialized: List[bool] = [False] * num_layers

    def _create_quantizer(self, dim, bits, device, dtype):
        if self.config.mode == "turbo_prod":
            return TurboQuantProd(dim, bits, self.config.rotation_seed, device, dtype, self.config.orthogonalize_jl)
        elif self.config.mode == "turbo_mse":
            return TurboQuantMSE(dim, bits, self.config.rotation_seed, device, dtype)
        elif self.config.mode == "qjl":
            return QJLQuantizer(dim, dim, self.config.rotation_seed, device, dtype, self.config.orthogonalize_jl)
        else:
            return TurboQuantProd(dim, bits, self.config.rotation_seed, device, dtype, self.config.orthogonalize_jl)

    def update(self, layer_idx: int, key_states: Tensor, value_states: Tensor):
        layer = self.layers[layer_idx]
        buf = self.buffers[layer_idx]

        if not self._initialized[layer_idx]:
            outlier_indices = self.outlier_detector.detect(key_states)
            layer.outlier_channel_indices = outlier_indices
            self._initialized[layer_idx] = True

        buf.add(key_states, value_states)

        if buf.has_overflow():
            overflow_keys, overflow_values = buf.flush_overflow()
            if overflow_keys is not None:
                self._quantize_and_store(layer_idx, overflow_keys, overflow_values)

    def _quantize_and_store(self, layer_idx: int, keys: Tensor, values: Tensor):
        layer = self.layers[layer_idx]
        outlier_indices = layer.outlier_channel_indices

        regular_keys, outlier_keys = self.outlier_detector.split_channels(keys, outlier_indices)

        reg_q = self.regular_quantizers[layer_idx]
        out_q = self.outlier_quantizers[layer_idx]

        if self.config.mode == "turbo_prod":
            reg_result = reg_q.quantize(regular_keys)
            out_result = out_q.quantize(outlier_keys)

            layer.quantized_key_indices = self._append_tensor(layer.quantized_key_indices, reg_result.mse_indices)
            layer.quantized_key_norms = self._append_tensor(layer.quantized_key_norms, reg_result.mse_norms)
            layer.quantized_key_qjl_bits = self._append_tensor(layer.quantized_key_qjl_bits, reg_result.qjl_bits)
            layer.quantized_key_residual_norms = self._append_tensor(layer.quantized_key_residual_norms, reg_result.residual_norms)

            layer.outlier_key_indices = self._append_tensor(layer.outlier_key_indices, out_result.mse_indices)
            layer.outlier_key_norms = self._append_tensor(layer.outlier_key_norms, out_result.mse_norms)
            layer.outlier_key_qjl_bits = self._append_tensor(layer.outlier_key_qjl_bits, out_result.qjl_bits)
            layer.outlier_key_residual_norms = self._append_tensor(layer.outlier_key_residual_norms, out_result.residual_norms)

        elif self.config.mode == "turbo_mse":
            reg_idx, reg_norms = reg_q.quantize(regular_keys)
            out_idx, out_norms = out_q.quantize(outlier_keys)

            layer.quantized_key_indices = self._append_tensor(layer.quantized_key_indices, reg_idx)
            layer.quantized_key_norms = self._append_tensor(layer.quantized_key_norms, reg_norms)
            layer.outlier_key_indices = self._append_tensor(layer.outlier_key_indices, out_idx)
            layer.outlier_key_norms = self._append_tensor(layer.outlier_key_norms, out_norms)

        elif self.config.mode == "qjl":
            reg_bits, reg_norms = reg_q.quantize(regular_keys)
            out_bits, out_norms = out_q.quantize(outlier_keys)

            layer.quantized_key_qjl_bits = self._append_tensor(layer.quantized_key_qjl_bits, reg_bits)
            layer.quantized_key_residual_norms = self._append_tensor(layer.quantized_key_residual_norms, reg_norms)
            layer.outlier_key_qjl_bits = self._append_tensor(layer.outlier_key_qjl_bits, out_bits)
            layer.outlier_key_residual_norms = self._append_tensor(layer.outlier_key_residual_norms, out_norms)

        v_quant, v_scale, v_zp = self.value_quantizer.quantize(values)
        layer.quantized_values = self._append_tensor(layer.quantized_values, v_quant)
        layer.value_scales = self._append_tensor(layer.value_scales, v_scale)
        layer.value_zero_points = self._append_tensor(layer.value_zero_points, v_zp)

        layer.quantized_seq_len += keys.shape[-2]

    def get_attention_scores(self, layer_idx: int, query_states: Tensor) -> Tensor:
        layer = self.layers[layer_idx]
        buf = self.buffers[layer_idx]
        scores_list = []

        if layer.quantized_seq_len > 0:
            quantized_scores = self._compute_quantized_scores(layer_idx, query_states)
            scores_list.append(quantized_scores)

        buffer_keys, _ = buf.get_buffer()
        if buffer_keys is not None and buffer_keys.shape[-2] > 0:
            buffer_scores = torch.matmul(query_states, buffer_keys.transpose(-1, -2))
            scores_list.append(buffer_scores)

        if not scores_list:
            return torch.zeros(
                query_states.shape[:-1] + (0,),
                device=query_states.device, dtype=query_states.dtype
            )

        return torch.cat(scores_list, dim=-1)

    def _compute_quantized_scores(self, layer_idx: int, query_states: Tensor) -> Tensor:
        layer = self.layers[layer_idx]
        outlier_indices = layer.outlier_channel_indices

        regular_q, outlier_q = self.outlier_detector.split_channels(query_states, outlier_indices)

        reg_quantizer = self.regular_quantizers[layer_idx]
        out_quantizer = self.outlier_quantizers[layer_idx]

        if self.config.mode == "turbo_prod":
            reg_quant = QuantizedKeyProd(
                layer.quantized_key_indices, layer.quantized_key_norms,
                layer.quantized_key_qjl_bits, layer.quantized_key_residual_norms
            )
            reg_scores = reg_quantizer.estimate_inner_product(regular_q, reg_quant)

            out_quant = QuantizedKeyProd(
                layer.outlier_key_indices, layer.outlier_key_norms,
                layer.outlier_key_qjl_bits, layer.outlier_key_residual_norms
            )
            out_scores = out_quantizer.estimate_inner_product(outlier_q, out_quant)

        elif self.config.mode == "turbo_mse":
            reg_scores = reg_quantizer.compute_rotated_score(
                regular_q, layer.quantized_key_indices, layer.quantized_key_norms
            )
            out_scores = out_quantizer.compute_rotated_score(
                outlier_q, layer.outlier_key_indices, layer.outlier_key_norms
            )

        elif self.config.mode == "qjl":
            reg_scores = reg_quantizer.estimate_inner_product(
                regular_q, layer.quantized_key_qjl_bits, layer.quantized_key_residual_norms
            )
            out_scores = out_quantizer.estimate_inner_product(
                outlier_q, layer.outlier_key_qjl_bits, layer.outlier_key_residual_norms
            )

        else:
            reg_quant = QuantizedKeyProd(
                layer.quantized_key_indices, layer.quantized_key_norms,
                layer.quantized_key_qjl_bits, layer.quantized_key_residual_norms
            )
            reg_scores = reg_quantizer.estimate_inner_product(regular_q, reg_quant)
            out_quant = QuantizedKeyProd(
                layer.outlier_key_indices, layer.outlier_key_norms,
                layer.outlier_key_qjl_bits, layer.outlier_key_residual_norms
            )
            out_scores = out_quantizer.estimate_inner_product(outlier_q, out_quant)

        return reg_scores + out_scores

    def get_value_output(self, layer_idx: int, attention_weights: Tensor) -> Tensor:
        layer = self.layers[layer_idx]
        buf = self.buffers[layer_idx]
        quant_len = layer.quantized_seq_len
        buf_len = buf.get_seq_len()

        outputs = []

        if quant_len > 0:
            quant_weights = attention_weights[..., :quant_len]
            values = self.value_quantizer.dequantize(
                layer.quantized_values, layer.value_scales, layer.value_zero_points
            )
            quant_output = torch.matmul(quant_weights, values)
            outputs.append(quant_output)

        if buf_len > 0:
            buf_weights = attention_weights[..., quant_len:quant_len + buf_len]
            _, buffer_values = buf.get_buffer()
            buf_output = torch.matmul(buf_weights, buffer_values)
            outputs.append(buf_output)

        if not outputs:
            return torch.zeros(
                attention_weights.shape[:-1] + (self.head_dim,),
                device=attention_weights.device, dtype=attention_weights.dtype
            )

        return sum(outputs)

    def get_seq_len(self, layer_idx: int) -> int:
        return self.layers[layer_idx].quantized_seq_len + self.buffers[layer_idx].get_seq_len()

    def _append_tensor(self, existing: Optional[Tensor], new: Tensor) -> Tensor:
        if existing is None:
            return new
        return torch.cat([existing, new], dim=-2 if new.dim() > 1 else -1)
