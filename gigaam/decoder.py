from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


class CTCHead(nn.Module):
    """
    Модуль CTC-головы для Connectionist Temporal Classification.
    """

    def __init__(self, feat_in: int, num_classes: int):
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(feat_in, num_classes, kernel_size=1)
        )

    def forward(self, encoder_output: Tensor) -> Tensor:
        return torch.nn.functional.log_softmax(
            self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
        )


class RNNTJoint(nn.Module):
    """
    Модуль совместной сети RNN-Transducer.
    Объединяет выходы энкодера и сети предсказания с помощью
    линейной трансформации,ReLU-активации и ещё одной линейной проекции.
    """

    def __init__(
        self, enc_hidden: int, pred_hidden: int, joint_hidden: int, num_classes: int
    ):
        super().__init__()
        self.enc_hidden = enc_hidden
        self.pred_hidden = pred_hidden
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(joint_hidden, num_classes))

    def joint(self, encoder_out: Tensor, decoder_out: Tensor) -> Tensor:
        """
        Объединяет выходы энкодера и сети предсказания в общее представление.
        """
        enc = self.enc(encoder_out).unsqueeze(2)
        pred = self.pred(decoder_out).unsqueeze(1)
        return self.joint_net(enc + pred).log_softmax(-1)

    def input_example(self, batch_size: int = 8) -> Tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        enc = torch.zeros(batch_size, self.enc_hidden, 1)
        dec = torch.zeros(batch_size, self.pred_hidden, 1)
        return enc.float().to(device), dec.float().to(device)

    def input_names(self) -> List[str]:
        return ["enc", "dec"]

    def output_names(self) -> List[str]:
        return ["joint"]

    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "enc": {0: "batch_size"},
            "dec": {0: "batch_size"},
            "joint": {0: "batch_size"},
        }

    def forward(self, enc: Tensor, dec: Tensor) -> Tensor:
        return self.joint(enc.transpose(1, 2), dec.transpose(1, 2))


class RNNTDecoder(nn.Module):
    """
    Модуль декодера RNN-Transducer.
    Отвечает за часть сети предсказания в архитектуре RNN-Transducer.
    """

    def __init__(self, pred_hidden: int, pred_rnn_layers: int, num_classes: int):
        super().__init__()
        self.blank_id = num_classes - 1
        self.pred_hidden = pred_hidden
        self.embed = nn.Embedding(num_classes, pred_hidden, padding_idx=self.blank_id)
        self.lstm = nn.LSTM(pred_hidden, pred_hidden, pred_rnn_layers)

    def predict(
        self,
        x: Optional[Tensor],
        state: Optional[Tensor],
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Делает предсказания на основе текущего входа и предыдущих состояний.
        Если вход не задан, в качестве начального входа используются нули.
        """
        if x is not None:
            emb: Tensor = self.embed(x)
        else:
            emb = torch.zeros(
                (batch_size, 1, self.pred_hidden), device=next(self.parameters()).device
            )
        g, hid = self.lstm(emb.transpose(0, 1), state)
        return g.transpose(0, 1), hid

    def input_example(self, batch_size: int = 8) -> Tuple[Tensor, Tensor, Tensor]:
        device = next(self.parameters()).device
        label = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
        hidden_h = torch.zeros(self.lstm.num_layers, batch_size, self.pred_hidden).to(
            device
        )
        hidden_c = torch.zeros(self.lstm.num_layers, batch_size, self.pred_hidden).to(
            device
        )
        return label, hidden_h, hidden_c

    def input_names(self) -> List[str]:
        return ["x", "hi", "ci"]

    def output_names(self) -> List[str]:
        return ["dec", "ho", "co"]

    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "x": {0: "batch_size"},
            "hi": {1: "batch_size"},
            "ci": {1: "batch_size"},
            "dec": {0: "batch_size"},
            "ho": {1: "batch_size"},
            "co": {1: "batch_size"},
        }

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Специфичный для ONNX forward: x, state = (h, c) -> x, h, c.
        """
        emb = self.embed(x)
        g, (h, c) = self.lstm(emb.transpose(0, 1), (h, c))
        return g.transpose(0, 1), h, c


class RNNTHead(nn.Module):
    """
    Модуль головы RNN-Transducer.
    Объединяет компоненты декодера и совместной сети архитектуры RNN-Transducer.
    """

    def __init__(self, decoder: Dict[str, int], joint: Dict[str, int]):
        super().__init__()
        self.decoder = RNNTDecoder(**decoder)
        self.joint = RNNTJoint(**joint)
