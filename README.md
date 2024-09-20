<p align="center">
      <br/>
            <img src="assets/spiderman-meme.jpg" alt="image" width="600" height="auto">
      <br/>
<p>
<p align="center"> 


<p align="center">
    <a href="https://www.trdavidson.com/self-recognition">
    <img alt="Blog post" src="https://img.shields.io/badge/blog-online-green">
    </a>
    <a href="https://www.python.org/downloads/release/python-3110/"><img alt="PyPi version" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://arxiv.org/abs/2407.06946">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-arXiv:2407.06946-b31b1b.svg">
    </a>
</p>

# Self-Recognition in Language Models.

This repository contains the official implementation for the paper 
_Self-Recognition in Language Models_ [[1]](#citation).
- 🎓 [full paper available on arXiv](https://arxiv.org/abs/2407.06946)
- 📝 [we also wrote a blog post](https://www.trdavidson.com/self-recognition)

---

## Overview
In our paper we proposed assessing self-recognition in language models (LMs) using model-generated security questions.
This approach takes three steps:
1. ❓ generate a set of questions;
2. 💬 generate a set of answers to these questions;
3. ⚖️ generate "verdicts" by showing LMs questions with n-answers, and prompting them to select their own.

This repository contains code to reproduce the experiments of the paper and is structured as follows:
```
.
├── src/
│   ├── models/ (currently support: Anthropic, Cohere, Google, Microsoft, OpenAI, TogetherAI)
│   ├── configs/ (configurations to create (i) questions, (ii) answers, (iii) verdicts)
│   └── *.py
├── data/
│   ├── api_settings/
│   ├── model_settings/
│   ├── prompts/
│   ├── questions/
│   ├── responses/
│   ├── verdicts/
│   └── llm_model_details.yaml
├── secrets.json (to be created)
└── gcp_secrets.json (optional)
```
A limited set of example questions, answers, and verdicts are provided in the `data/` directory.

### Usage Steps
We use `hydra` to manage configurations. 
The main entry point is `src/run.py`, which takes a configuration file as input.
Configuration files are stored in `src/configs/` and are used to generate questions, answers, and verdicts:
```
src/
└── configs/
    ├── generate_questions.yaml
    ├── generate_responses.yaml
    └── generate_verdicts.yaml
```
Simply navigate to these files to specify the model(s) you want to use. LM wrappers for most leading
providers are included in `src/models/`.

After generating questions, responses, and verdicts, `hydra` will save the output to a specified directory, `logs/` per default.

**questions**: 
- to generate: `python src/run.py defaults.experiments=generate_questions`
- saves a `questions.csv` file to `logs/<your-experiment>`
- copy this file to `data/questions/` for the next step

**responses**:
- to generate: `python src/run.py defaults.experiments=generate_responses`
- saves a `responses.csv` file to `logs/<your-experiment>`
- copy this file to `data/responses/` for the next step

**verdicts**:
- to generate: `python src/run.py defaults.experiments=generate_verdicts`
- saves a `verdicts.csv` file to `logs/<your-experiment>`
- copy this file to `data/verdicts/` for the next step

**evaluation**:
- to process verdicts and make sure they are correctly formatted: `python src/verdict_evaluation.py --base_folder=<path-to-verdicts>`
- this creates `verdicts_extracted.csv` in the same directory
- to evaluate the performance of the model: `python src/evaluations.py --base_folder=<path-to-extacted-verdicts>`

Having run these steps, you can use various tools to analyze the results. For example, see the files:
- `src/analys.py`
- `src/visualization.py`



## Setup
The simplest way to get started is to:
1. clone this repository, then
2. create a `secrets.json` file in the root directory with the following structure:
```json
{
    "openai": {
        "api_key": "<your-key>"
    },
    "azure": {
        "api_key": "<your-key>"
    },
    "anthropic": {
        "api_key": "<your-key>"
    },
    "google": {
        "api_key": "<your-key>"
    },
    "cohere": {
        "api_key": "<your-key>"
    },
  "together_ai": {
      "api_key": "<your-key>"
  }
}
```
In this file, insert your own API key for one of the following providers: 
{Anthropic, Cohere, Google, OpenAI, Microsoft}. This `secrets.json` file is part of the `.gitignore`, to prevent you 
from accidentally pushing your raw keys to GitHub :). (see 'note' below if using Google/Azure models)

Next, create a virtual environment and install the packages listed in `requirements.txt`. Once this is done you're all 
set.

For any questions, feel free to open a ticket or reach out directly to [Tim](tim.davidson@epfl.ch) :).


### Note on Google/MSFT Azure
If you are using Google or MSFT Azure, you also need to update the relevant endpoints in 
`data/api_settings/apis.yaml`. At the time of release, the Google Vertex API does not support simple API keys in all 
regions. To get around this, you have to create a (1) service account, (2) set some permissions, (3) download a .json. 
Save the exported .json file in a file called `gcp_secrets.json` in the root directory of this project 
(also in `.gitignore`). 
[See the following docs for a walkthrough](https://cloud.google.com/iam/docs/service-accounts-create).

## License
MIT

## Citation
Please cite our work using one of the following if you end up using the repository - thanks!

```
[1] T.R. Davidson, V. Surkov, V. Veselovsky, G. Russo, R. West, C. Gulcehre. 
Self-Recognition in Language Models. arXiv preprint, arXiv:2407.06946, 2024.
```

BibTeX format:
```
@article{davidson2024selfrecognitionlanguagemodels,
      title={Self-Recognition in Language Models}, 
      author={Tim R. Davidson and 
              Viacheslav Surkov and 
              Veniamin Veselovsky and 
              Giuseppe Russo and 
              Robert West and 
              Caglar Gulcehre},
      year={2024},
      journal={EMNLP},
      url={https://arxiv.org/abs/2407.06946}
}
```
