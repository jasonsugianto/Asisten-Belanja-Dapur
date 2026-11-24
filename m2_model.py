"""M2: arsitektur wav2vec2 + BiLSTM + CTC head (target FONEM) dan vocab fonem.

Jalur akustik (pilihan A): fitur wav2vec2 -> BiLSTM kecil -> Linear -> CTC.
Target CTC = fonem IPA dari g2p_id (sama seperti M1), supaya pembanding adil.

Butuh: torch, transformers, g2p_id_py
"""
import torch.nn as nn
from transformers import Wav2Vec2Model

from data import load_script
from nlu_phoneme import to_phonemes  # g2p yang sama dengan M1 (tuple fonem datar)

# --- blank CTC di indeks 0; sisanya fonem ---
BLANK = "<blank>"


def build_vocab():
    """Kumpulkan semua fonem yang muncul di naskah -> {fonem: id}, blank di 0.
    Vocab ditentukan dari script.csv agar deterministik dan tidak bergantung urutan file."""
    phones = set()
    for s in load_script().values():
        phones.update(to_phonemes(s["transcript"]))
    phones.discard("")
    vocab = [BLANK] + sorted(phones)
    return {p: i for i, p in enumerate(vocab)}


def transcript_to_ids(text, vocab):
    """Teks -> daftar id fonem (buang yang tak ada di vocab, jarang terjadi)."""
    return [vocab[p] for p in to_phonemes(text) if p in vocab]


class Wav2Vec2BiLSTMCTC(nn.Module):
    """wav2vec2-base (fitur akustik) -> BiLSTM kecil -> proyeksi -> logit CTC fonem."""

    def __init__(self, vocab_size, base_model="facebook/wav2vec2-base",
                 lstm_hidden=128, lstm_layers=1, freeze_feature_encoder=True):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(base_model)
        if freeze_feature_encoder:
            # bekukan CNN feature extractor: wajar untuk data kecil, mempercepat & menstabilkan
            self.w2v.feature_extractor._freeze_parameters()
        hid = self.w2v.config.hidden_size
        self.bilstm = nn.LSTM(hid, lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(2 * lstm_hidden, vocab_size)

    def forward(self, input_values, attention_mask=None):
        """input_values: (B, T) float32 @16kHz. Kembalikan log_probs (T', B, V) untuk CTC."""
        feat = self.w2v(input_values, attention_mask=attention_mask).last_hidden_state  # (B, T', H)
        seq, _ = self.bilstm(feat)                       # (B, T', 2*hidden)
        logits = self.head(self.dropout(seq))            # (B, T', V)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.transpose(0, 1)                 # (T', B, V) <- urutan yang diminta CTCLoss

    def feat_lengths(self, input_lengths):
        """Panjang keluaran wav2vec2 untuk tiap panjang input (untuk CTC input_lengths)."""
        return self.w2v._get_feat_extract_output_lengths(input_lengths).long()


def greedy_decode(log_probs_tbv, id2phone, blank=0):
    """CTC greedy collapse. log_probs_tbv: (T', B, V). Kembalikan list[list[str fonem]] per batch."""
    ids = log_probs_tbv.argmax(dim=-1).transpose(0, 1).tolist()   # (B, T')
    outs = []
    for seq in ids:
        prev, phones = None, []
        for i in seq:
            if i != blank and i != prev:
                phones.append(id2phone[i])
            prev = i
        outs.append(phones)
    return outs
