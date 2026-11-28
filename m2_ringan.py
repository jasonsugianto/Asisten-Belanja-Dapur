"""Pemercepat M2 (wav2vec2 + CTC) di CPU: (1) pangkas lapisan transformer, (2) kuantisasi int8 dinamis."""
import copy
import torch


def potong_lapisan(model, k):
    """Pertahankan k lapisan transformer pertama (dari 24 pada XLS-R large). Perlu di-fine-tune ulang."""
    enc = model.wav2vec2.encoder
    if k < len(enc.layers):
        enc.layers = enc.layers[:k]
        model.config.num_hidden_layers = k
    return model


def int8(model, inplace=False):
    """Lapisan Linear dikuantisasi int8 (konvolusi tetap fp32). inplace=False membuat salinan CPU baru."""
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:                                       # torch lama
        from torch.quantization import quantize_dynamic
    base = model.cpu().eval() if inplace else copy.deepcopy(model).cpu().eval()
    return quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8, inplace=inplace)


def konfigurasi_xlsr_large(vocab=40):
    """Arsitektur sama dengan XLS-R/XLSR-53 large (bobot acak; hanya untuk mengukur kecepatan)."""
    from transformers import Wav2Vec2Config
    return Wav2Vec2Config(vocab_size=vocab, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                          intermediate_size=4096, feat_extract_norm="layer", do_stable_layer_norm=True,
                          pad_token_id=0, mask_time_prob=0.0, layerdrop=0.0)
