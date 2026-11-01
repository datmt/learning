"""Model definition + preprocessing shared by export and serving. Must match the CRNN cell in train.ipynb."""
import string
import numpy as np
import cv2

IMG_H, IMG_W = 80, 384
CHARS = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~"
NCLS = len(CHARS) + 1          # index 0 = CTC blank


def preprocess(gray):
    """uint8 HxW -> float32 1xHxW, per-image standardised. Replicates the notebook exactly: load_gray (resize to 2x width) then to_input."""
    gray = cv2.resize(gray, (IMG_W * 2, int(IMG_W * 2 / 3.1)), interpolation=cv2.INTER_AREA)
    g = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((g - g.mean()) / (g.std() + 1e-3))[None]


def ctc_decode(logits):
    """logits: B x T x NCLS (numpy) -> list of (text, mean confidence of emitted characters)."""
    out = []
    p = logits - logits.max(-1, keepdims=True); p = np.exp(p); p /= p.sum(-1, keepdims=True)
    for lg, pr in zip(logits.argmax(-1), p):
        s, conf, prev = [], [], 0
        for t, k in enumerate(lg):
            if k != prev and k != 0: s.append(CHARS[k - 1]); conf.append(float(pr[t, k]))
            prev = k
        out.append(("".join(s), float(np.mean(conf)) if conf else 0.0))
    return out


def build_torch_model():
    import torch.nn as nn
    def blk(i, o): return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(*blk(1, 32), nn.MaxPool2d(2), *blk(32, 64), nn.MaxPool2d(2), *blk(64, 128), *blk(128, 128), nn.MaxPool2d((2, 1)),
                                     *blk(128, 256), *blk(256, 256), nn.MaxPool2d((2, 1)))
            self.proj = nn.Linear(256 * (IMG_H // 16), 256); self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=.2)
            self.out = nn.Linear(512, NCLS)
        def forward(self, x):
            f = self.cnn(x); b, c, h, w = f.shape
            f = self.proj(f.permute(0, 3, 1, 2).reshape(b, w, c * h)); f, _ = self.rnn(f); return self.out(f)
    return CRNN()


def ctc_decode_detail(logits):
    """Single sample T x NCLS -> (text, [per-character confidence])."""
    p = logits - logits.max(-1, keepdims=True); p = np.exp(p); p /= p.sum(-1, keepdims=True)
    s, conf, prev = [], [], 0
    for t, k in enumerate(logits.argmax(-1)):
        if k != prev and k != 0: s.append(CHARS[k - 1]); conf.append(float(p[t, k]))
        prev = k
    return "".join(s), conf
