---
license: cc-by-nc-4.0
datasets:
- amphion/Emilia-Dataset
pipeline_tag: text-to-speech
---

# *Metis*: A Foundation Speech Generation Model with Masked Generative Pre-training

[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/pdf/2502.03128)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/metis)
[![readme](https://img.shields.io/badge/README-GitHub-blue)](https://github.com/open-mmlab/Amphion/models/tts/metis/README.md)
<!-- [![ModelScope](https://img.shields.io/badge/ModelScope-model-cyan)](https://modelscope.cn/models/amphion/metis)
 -->

## Overview

We introduce ***Metis***, a foundation model for unified speech generation.
Unlike previous task-specific or multi-task models, Metis follows a pre-training and fine-tuning paradigm. It is pre-trained on large-scale unlabeled speech data using masked generative modeling and then fine-tuned to adapt to diverse speech generation tasks.
Specifically, (1) Metis utilizes two discrete speech representations: SSL tokens derived from speech self-supervised learning (SSL) features, and acoustic tokens directly quantized from waveforms. (2) Metis performs masked generative pre-training on SSL tokens, utilizing 300K hours of diverse speech data, without any additional condition. (3) Through fine-tuning with task-specific conditions, Metis achieves efficient adaptation to various speech generation tasks while supporting multimodal input, even when using limited data and trainable parameters.
Experiments demonstrate that Metis can serve as a foundation model for unified speech generation: Metis outperforms state-of-the-art task-specific or multi-task systems
across five speech generation tasks, including zero-shot text-to-speech, voice conversion, target speaker extraction, speech enhancement, and lip-to-speech, even with fewer than 20M trainable parameters or 300 times less training data.
Audio samples are are available at [demo page](https://metis-demo.github.io/).

<!-- ## News

- **2025/02/25**: We release ***Metis***, a foundation model for unified speech generation. The system supports zero-shot text-to-speech, voice conversion, target speaker extraction, speech enhancement, and lip-to-speech.

 -->

## Model Introduction

Metis is fully compatible with MaskGCT and shares several key model components with it. These shared components are:


| Model Name                                                                        | Description                                                                            |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [Semantic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec) | Converting speech to semantic tokens.                                                  |
| [Acoustic Codec](https://huggingface.co/amphion/MaskGCT/tree/main/acoustic_codec) | Converting speech to acoustic tokens and reconstructing waveform from acoustic tokens. |
| [Semantic2Acoustic](https://huggingface.co/amphion/MaskGCT/tree/main/s2a_model)         | Predicts acoustic tokens conditioned on semantic tokens.    

We open-source the pretrained model checkpoint of the first stage of Metis (with masked generative pre-training), as well as the fine-tuned models for speech enhancement (SE), target speaker extraction (TSE), voice conversion (VC), lip-to-speech (L2S), and the unified multi-task (Omni) model.

For zero-shot text-to-speech, you can download the text2semantic model from MaskGCT, which is compatible with the Metis framework.

| Model Name | Description |
| --- | --- |
| [Metis-Base](https://huggingface.co/amphion/metis/tree/main/metis_base) | The base model pre-trained with masked generative pre-training. |
| [Metis-TSE](https://huggingface.co/amphion/metis/tree/main/metis_tse) | Fine-tuned model for target speaker extraction. Available in both full-scale and LoRA (r = 32) versions. |
| [Metis-VC](https://huggingface.co/amphion/metis/tree/main/metis_vc) | Fine-tuned model for voice conversion. Available in full-scale version. |
| [Metis-SE](https://huggingface.co/amphion/metis/tree/main/metis_se) | Fine-tuned model for speech enhancement. Available in both full-scale and LoRA (r = 32) versions. |
| [Metis-L2S](https://huggingface.co/amphion/metis/tree/main/metis_l2s) | Fine-tuned model for lip-to-speech. Available in full-scale version. |
| [Metis-TTS](https://huggingface.co/amphion/MaskGCT/tree/main/t2s_model) | Zero-shot text-to-speech model (as same as the first stage of MaskGCT). |
| [Metis-Omni](https://huggingface.co/amphion/metis/tree/main/metis_omni) | Unified multi-task model supporting zero-shot TTS, VC, TSE, and SE. |


## Usage

## Citations 

If you use Metis in your research, please cite the following paper:

```bibtex
@article{wang2025metis,
  title={Metis: A Foundation Speech Generation Model with Masked Generative Pre-training},
  author={Wang, Yuancheng and Zheng, Jiachen and Zhang, Junan and Zhang, Xueyao and Liao, Huan and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2502.03128},
  year={2025}
}
@inproceedings{wang2024maskgct,
  author={Wang, Yuancheng and Zhan, Haoyue and Liu, Liwei and Zeng, Ruihong and Guo, Haotian and Zheng, Jiachen and Zhang, Qiang and Zhang, Xueyao and Zhang, Shunsi and Wu, Zhizheng},
  title={MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer},
  booktitle    = {{ICLR}},
  publisher    = {OpenReview.net},
  year         = {2025}
}
@article{amphion_v0.2,
  title        = {Overview of the Amphion Toolkit (v0.2)},
  author       = {Jiaqi Li and Xueyao Zhang and Yuancheng Wang and Haorui He and Chaoren Wang and Li Wang and Huan Liao and Junyi Ao and Zeyu Xie and Yiqiao Huang and Junan Zhang and Zhizheng Wu},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2501.15442},
}
@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```