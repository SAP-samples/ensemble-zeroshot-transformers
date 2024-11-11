# Plug, Play, and Fuse: Zero-Shot Joint Decoding via Word-Level Re-ranking Across Diverse Vocabularies (WMT 2024)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2408.11327-b31b1b.svg)](https://arxiv.org/abs/2408.11327)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/ensemble-zeroshot-transformers)](https://api.reuse.software/info/github.com/SAP-samples/ensemble-zeroshot-transformers)

## Description
This repository contains zero-shot joint decoding sample code of our [paper](https://arxiv.org/abs/2408.11327) for models hosted on the [huggingface hub](https://huggingface.co/) based on [transformers](https://github.com/huggingface/transformers) library.

Specifically, we provide scripts to reproduce results in **Table 3** of the paper by combing Speech Translation Seamless model and Text-based Machine Translation Madlad model. Further, we provide sample files to re-rank N-best list as well. If you would like to use other models, the code can be adapted with slight changes.

## Abstract

Recent advancements in NLP have resulted in models with specialized strengths, such as processing multimodal inputs or excelling in specific domains. However, real-world tasks, like multimodal translation, often require a combination of these strengths, such as handling both translation and image processing. While individual translation and vision models are powerful, they typically lack the ability to perform both tasks in a single system. Combining these models poses challenges, particularly due to differences in their vocabularies, which limit the effectiveness of traditional ensemble methods to post-generation techniques like Nbest list re-ranking. In this work, we propose a novel zero-shot ensembling strategy that allows for the integration of different models during the decoding phase without the need for additional training. Our approach re-ranks beams during decoding by combining scores at the word level, using heuristics to predict when a word is completed. We demonstrate the effectiveness of this method in machine translation scenarios, showing that it enables the generation of translations that are both speechand image-aware while also improving overall translation quality.

## Requirements
We only rely on the transformers library for fusing models. However, to make the inference possible on 1 GPU and load fine-tuned models, peft and bitsandbytes are also required.

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Peft](https://github.com/huggingface/peft)
- [Bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

## Download and Installation

Only installing the above libraries is necessary. No other installation is required except cloning the git repository.

Clone this repository
```
git clone https://github.com/SAP-samples/ensemble-zeroshot-transformers
cd ensemble-zeroshot-transformers
```
## Usage

We provide scripts for 3 scenarios

1. **Generation from individual models**
   ```
   python madlad_generate.py --input_file ${source_text} --output_file ${output_file}
   ```

   The default parameters assumes the language direction used in the paper. If you would like to change the direction, adapt the script in few places with the right language id according to madlad configuration

   For Seamless, use ```seamless_generate.py``` and provide a tsv in the format of MuST-SHE data. If your data is in different format, change the lines 80 to 84 to read your own data format.

2. **Re-ranking N-best list**

   By default in the *_generate.py scripts, we save a N-best list size equal to the number of beams and their corresponding scores. This can be changed if required.

   This N-best list hypotheses and scores can be passed to *_ranknbestlist.py script for re-ranking with a different model and generating combined score.

   You can use the script in this way

   ```
   python seamless_ranknbestlist.py --wav_input ${tsv_file} --text_input ${nbestlist.txt} --score_output ${score_output.txt}
   ```

   Similarly, few parameters are set to default values for our experiments and need to be changed if you would like to try different models.

3. **Joint Decoding via word-level re-ranking**

   To use Madlad as Generator and Seamless as re-ranker, we extend the [logits_processor](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py) where we extract the top candidates, estimate probability with Seamless and merge the scores.

   As described in the paper, you can control the alpha or topk parameter for your setting. Note that increasing topk would need more memory.

   ```
   python madlad_seamless_decode.py --input_file ${input_txt} --wav_file ${wav_input} --output_file ${output_file} --topk 5 --num_beams 5 --alpha 0.8 --beta 0.2
   ```

   This script uses the logits processor in madlad_seamless_logits_processor.py to do the re-ranking. While we wanted to implement the code model agnostic, the current version only supports Seamless model as each model needs specific tokens in the prefix among few other stuff. If you would like to adapt for other model, please ask us and we will be glad to help you

## Known Issues
1. The decoding time is too slow!!!
   We currently do not have KV-Cached implementation in the re-ranking stage making it extremely slower and are not planning to. If one would like to collaborate and can provide insights, we would be happy to recieve your help!


## Citations
If you use this code in your research or want to refer to our work, please cite:

```
@InProceedings{koneru-EtAl:2024:WMT,
  author    = {Koneru, Sai  and  Huck, Matthias  and  Exel, Miriam  and  Niehues, Jan},
  title     = {Plug, Play, and Fuse: Zero-Shot Joint Decoding via Word-Level Re-ranking across Diverse Vocabularies},
  booktitle      = {Proceedings of the Ninth Conference on Machine Translation},
  month          = {November},
  year           = {2024},
  address        = {Miami, Florida, USA},
  publisher      = {Association for Computational Linguistics},
  pages     = {1467--1481},
  abstract  = {Recent advancements in NLP have resulted in models with specialized strengths, such as processing multimodal inputs or excelling in specific domains. However, real-world tasks, like multimodal translation, often require a combination of these strengths, such as handling both translation and image processing. While individual translation and vision models are powerful, they typically lack the ability to perform both tasks in a single system. Combining these models poses challenges, particularly due to differences in their vocabularies, which limit the effectiveness of traditional ensemble methods to post-generation techniques like N-best list re-ranking. In this work, we propose a novel zero-shot ensembling strategy that allows for the integration of different models during the decoding phase without the need for additional training. Our approach re-ranks beams during decoding by combining scores at the word level, using heuristics to predict when a word is completed. We demonstrate the effectiveness of this method in machine translation scenarios, showing that it enables the generation of translations that are both speech- and image-aware while also improving overall translation quality.},
  url       = {https://aclanthology.org/2024.wmt-1.133}
}
```
#### Authors:
 - Sai Koneru
 - Matthias Huck
 - Miriam Exel
 - Jan Niehues


## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
