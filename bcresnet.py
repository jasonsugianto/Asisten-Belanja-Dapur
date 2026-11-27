"""BC-ResNet versi sederhana (Kim et al., 2021) untuk klasifikasi kata pendek.
Catatan jujur: ini implementasi ringkas dari ide intinya (konvolusi frekuensi 3x1 + konvolusi waktu
1x3 teredam-dilasi yang disiarkan/broadcast ke seluruh frekuensi). SubSpectralNorm TIDAK dipakai,
jadi bukan reproduksi persis dari paper. Tanpa bobot terlatih (dilatih dari nol)."""
import torch
import torch.nn as nn

from m3_utils import mel_filterbank


class LogMel(nn.Module):
    """Gelombang [B,N] -> log-mel [B,1,40,T], dinormalisasi per jendela."""
    def __init__(self, n_mels=40):
        super().__init__()
        self.register_buffer("win", torch.hann_window(400))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(n_mels=n_mels)))

    def forward(self, wav):
        spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400, window=self.win,
                          center=True, return_complex=True).abs() ** 2        # [B,257,T]
        mel = torch.matmul(self.fb, spec)                                      # [B,40,T]
        x = torch.log(mel + 1e-6)
        m = x.mean(dim=(1, 2), keepdim=True)
        s = x.std(dim=(1, 2), keepdim=True)
        return ((x - m) / (s + 1e-5)).unsqueeze(1)


class BCResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        self.transition = cin != cout or stride != 1
        self.pre = (nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
                    if cin != cout else nn.Identity())
        self.f2 = nn.Sequential(                                   # frekuensi: depthwise 3x1
            nn.Conv2d(cout, cout, (3, 1), stride=(stride, 1), padding=(1, 0), groups=cout, bias=False),
            nn.BatchNorm2d(cout))
        self.f1 = nn.Sequential(                                   # waktu: depthwise 1x3 (dilasi)
            nn.Conv2d(cout, cout, (1, 3), padding=(0, dilation), dilation=(1, dilation), groups=cout, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 1, bias=False), nn.Dropout2d(dropout))
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pre(x)
        y = self.f2(x)                                             # [B,C,F',T]
        z = self.f1(y.mean(dim=2, keepdim=True))                   # [B,C,1,T]  (rata-rata frekuensi)
        out = y + z                                                # z disiarkan ke seluruh frekuensi
        if not self.transition:
            out = out + x                                          # skip identitas hanya pada blok normal
        return self.act(out)


class BCResNet(nn.Module):
    def __init__(self, n_classes, tau=2, n_mels=40):
        super().__init__()
        c = [int(8 * tau), int(12 * tau), int(16 * tau), int(20 * tau)]
        n, s, d = [2, 2, 4, 4], [1, 2, 2, 1], [1, 2, 4, 8]
        self.front = LogMel(n_mels)
        self.stem = nn.Sequential(nn.Conv2d(1, 16 * tau if isinstance(tau, int) else int(16 * tau), 5,
                                            stride=(2, 1), padding=2, bias=False),
                                  nn.BatchNorm2d(16 * tau if isinstance(tau, int) else int(16 * tau)), nn.ReLU())
        layers, cin = [], int(16 * tau)
        for i in range(4):
            for j in range(n[i]):
                layers.append(BCResBlock(cin, c[i], stride=s[i] if j == 0 else 1, dilation=d[i]))
                cin = c[i]
        self.blocks = nn.Sequential(*layers)
        fdim = n_mels // 2                                         # setelah stem (stride 2)
        for st in s:
            fdim = (fdim + 2 * 1 - 3) // st + 1 if st > 1 else fdim
        self.head = nn.Sequential(
            nn.Conv2d(cin, cin, (fdim, 5), padding=(0, 2), groups=cin, bias=False),
            nn.Conv2d(cin, int(32 * tau), 1, bias=False), nn.BatchNorm2d(int(32 * tau)), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(int(32 * tau), n_classes, 1))

    def forward(self, wav):
        x = self.blocks(self.stem(self.front(wav)))
        return self.head(x).flatten(1)                             # logit [B,n_classes]
