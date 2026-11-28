"""Export the aligned OCR checkpoint for Ojas's native ONNX executor."""

import argparse
from pathlib import Path
import sys

import onnx
import torch
from torch import nn


class SpatialConv(nn.Module):
    """Represent a width-only Conv1d with the equivalent height-one Conv2d."""

    def __init__(self, layer):
        super().__init__()
        self.conv = nn.Conv2d(
            layer.in_channels, layer.out_channels, (1, layer.kernel_size[0]),
            stride=(1, layer.stride[0]), padding=(0, layer.padding[0]),
            dilation=(1, layer.dilation[0]), groups=layer.groups,
            bias=layer.bias is not None,
        )
        with torch.no_grad():
            self.conv.weight.copy_(layer.weight.unsqueeze(2))
            if layer.bias is not None:
                self.conv.bias.copy_(layer.bias)

    def forward(self, inputs):
        return self.conv(inputs.unsqueeze(2)).squeeze(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    from ocr_aligned import AlignedOCR, PREDICTION_SHIFTS

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = AlignedOCR()
    model.load_state_dict(checkpoint["weights"])
    model.column_features[0] = SpatialConv(model.column_features[0])
    model.alignment.local = SpatialConv(model.alignment.local)
    model.alignment.context[0] = SpatialConv(model.alignment.context[0])

    class OjasOCR(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            threshold = checkpoint.get("shift_review_threshold")
            self.register_buffer("cutoff", torch.tensor([1.01 if threshold is None else threshold]))

        def forward(self, patches):
            features = self.model.encoder(patches).squeeze(2)
            columns = self.model.column_features(features)
            glyphs, _ = self.model.alignment(columns)
            probabilities = self.model.classifier(glyphs).softmax(-1)
            return probabilities.mean(0, keepdim=True), self.cutoff

    example = torch.zeros((len(PREDICTION_SHIFTS), 16, 12, 50))
    torch.onnx.export(
        OjasOCR().eval(), (example,), args.output, dynamo=False, opset_version=17,
        input_names=["shifted_patches"], output_names=["probabilities", "review_threshold"],
    )
    onnx.checker.check_model(args.output)


if __name__ == "__main__":
    main()
