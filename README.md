<div align="center">
<img src="../TrustLLM/asserts/logo.png" >

<p align="center">
   <a href="https://atlas.nomic.ai/map/2588798c-fd96-42db-8ad4-c94dd5d4daed/97226b7c-2ae2-457a-9ff0-53ba96659ee2?xs=-29.43858&xf=41.60393&ys=-43.87149&yf=43.64643" target="_blank">🌐 Dataset</a> | <a href="" target="_blank">📃 Paper </a> | <a href="https://github.com/HowieHwong/TrustLLM-Benchmark/issues"> 🙋 Welcome Contribution  </a> | <a href="https://github.com/HowieHwong/TrustLLM-Benchmark/blob/master/LICENSE"> 📜 License</a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E.svg?style=flat-square&logo=JavaScript&logoColor=black" alt="JavaScript" />
<img src="https://img.shields.io/badge/HTML5-E34F26.svg?style=flat-square&logo=HTML5&logoColor=white" alt="HTML5" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat-square&logo=JSON&logoColor=white" alt="JSON" />
</p>
<img src="https://img.shields.io/github/last-commit/HowieHwong/TrustLLM-Benchmark?style=flat-square&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/HowieHwong/TrustLLM-Benchmark?style=flat-square&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/HowieHwong/TrustLLM-Benchmark?style=flat-square&color=5D6D7E" alt="GitHub top language" />
</div>


## Introduction

Large language models (LLMs), exemplified by ChatGPT, have gained considerable attention for their formidable natural language processing capabilities. Nonetheless, these LLMs present many challenges, particularly in the realm of trustworthiness. Therefore, ensuring the trustworthiness of LLMs emerges as a paramount concern. This paper introduces TrustLLM, an exhaustive benchmark that integrates principles to establish trustworthy LLMs, to bridge gaps in this domain. 
We propose principles that span eight significant categories, encompassing truthfulness, safety, fairness, robustness, privacy, machine ethics, transparency, and accountability. Based on these principles, we establish a benchmark across six domains. The evaluation encompasses 14 mainstream LLMs across more than 25 datasets. 
Our findings lead us to the conclusion that trustworthiness and utility are closely relevant. Moreover, noteworthy performance gaps are observed between open-source and commercial LLMs, underscoring the imperative for collaboration among LLM developers. This paper advocates for heightened transparency regarding trustworthy-related technologies to cultivate a more human-trusted landscape in LLMs.

## Models

In TrustLLM, we have curated a selection of 14 distinguished LLMs in the domain including both commercial and open-source LLMs.

<img src="assets/models_overview.png" >

We are still working on including more LLMs.

## Dataset

### Overview

We have released the dataset in <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset" target="_blank">huggingface</a>.

<img src="assets/dataset_overview.png" >

### Raw dataset

| Dataset       | Source                                                        | Dataset   | Source   |
|---------------|---------------------------------------------------------------|------|------|
| SQuAD2.0 | <a href="https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/">Website</a> | CODAH  | <a href="https://github.com/Websail-NU/CODAH">Website</a>  |
| HotpotQA           | <a href="https://hotpotqa.github.io/">Website</a>                                                           | AdversarialQA  | <a href="https://adversarialqa.github.io/">Website</a>  |
| Climate-FEVER           | <a href="https://huggingface.co/datasets/climate_fever">Website</a>                                                          | SciFact | <a href="https://allenai.org/data/scifact">Website</a> |
| COVID-Fact         | <a href="https://github.com/asaakyan/covidfact">Website</a>                                                          | HealthVER | <a href="https://github.com/sarrouti/HealthVer">Website</a>|
| TruthfulQA         | <a href="https://github.com/sylinrl/TruthfulQA">Website</a>                                                          | HaluEval | <a href="https://github.com/RUCAIBox/HaluEval">Website</a>|
| Sycophancy         | <a href="https://github.com/nrimsky/LM-exp/blob/main/datasets/sycophancy/sycophancy.json">Website</a>                                                          | TBD | <a href="TBD">Website</a>|
| WinoBias         | <a href="https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino">Website</a>        | StereoSet | <a href="https://github.com/moinnadeem/StereoSet">Website</a>|



## Run Evaluation




---
## Submit Your Result

- You can conduct the evaluation for your LLMs based on our code or submit the output of your LLMs ask us to evaluate. 

- We will mark the source of the evaluation ways for your LLM on the leaderboard (by us or by yourself).

If you would like us to conduct the evaluation for you, you will need to first download our dataset in <a href="https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset" target="_blank">huggingface</a>. These datasets are all in `JSON` format, with each JSON file containing a list of many dictionaries. Each dictionary has a fixed key `prompt`. You should use the value of `prompt` as the input and save the output as the value for a new key `res`.

For example:

```python
import json

filename = 'DATASET_FILEPATH'

with open(filename, 'r') as f:
    data = json.load(f)

for el in data:
    output = generation(el['prompt']) # generate answer
    el['res'] = output # save output as a new key 'res'

with open(filename, 'w') as f:
    json.dumps(data, f, indent=4)
```

Finally, you need to compress your files as a `zip.` file according to the following directory structure:

```sh
└── Results/
    ├── Safety/
    │   ├── filename1.json
    │   ├── filename2.json
    │   ├── filename3.json
    │   └── ...
    ├── Truthfulness/
    │   ├── filename1.json
    │   ├── filename2.json
    │   └── ...
    ├── Fairness
    ├── Robustness
    ├── Privacy
    └── Machine Ethics

```

Send it to the email: `trustllm@gmail.com`


## TODO




## Contact

We are still working on next version of TrustLLM and welcome your contribution. You can contribute to the following sections:

- Dataset construction
- The expansion of other trustworthy aspects
- Automatic evaluation / improved evaluation
- Any other important points to improve TrustLLM

Email: `howiehwong@gmail.com`

## Citation



## Stargazers over time




## License

TrustLLM is licensed under the terms of the MIT license. See LICENSE for more details.
