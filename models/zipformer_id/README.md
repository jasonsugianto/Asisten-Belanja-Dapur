---
language: id
license: apache-2.0
tags:
  - icefall
  - sherpa-onnx
  - phoneme-recognition
  - automatic-speech-recognition
datasets:
  - mozilla-foundation/common_voice_13_0
  - indonesian-nlp/librivox-indonesia
  - google/fleurs
---

# Sherpa-onnx Pruned Stateless Zipformer RNN-T Streaming ID

Sherpa-onnx Pruned Stateless Zipformer RNN-T Streaming ID is an automatic speech recognition model trained on the following datasets:

- [Common Voice ID](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)
- [LibriVox Indonesia](https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia)
- [FLEURS ID](https://huggingface.co/datasets/google/fleurs)

Instead of being trained to predict sequences of words, this model was trained to predict sequence of phonemes, e.g. `['p', 'ə', 'r', 'b', 'u', 'a', 't', 'a', 'n', 'ɲ', 'a']`. Therefore, the model's [vocabulary](https://huggingface.co/bookbot/pruned-transducer-stateless7-streaming-id/blob/main/data/lang_phone/tokens.txt) contains the different IPA phonemes found in [g2p ID](https://github.com/bookbot-kids/g2p_id).

This model was converted from the TorchScript version of [Pruned Stateless Zipformer RNN-T Streaming ID](https://huggingface.co/bookbot/pruned-transducer-stateless7-streaming-id) to ONNX format.

## Converting from TorchScript

Refer to the [official instructions](https://icefall.readthedocs.io/en/latest/model-export/export-onnx.html#export-the-model-to-onnx) for conversion to ONNX.

## Frameworks

- [k2](https://github.com/k2-fsa/k2)
- [icefall](https://github.com/bookbot-hive/icefall)
- [lhotse](https://github.com/bookbot-hive/lhotse)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [onnx](https://github.com/onnx/onnx)
- [onnxruntime](https://github.com/microsoft/onnxruntime)