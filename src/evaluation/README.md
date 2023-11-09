# 🚀 Evaluation Scripts

---

## Truthfulness

---

## Safety

#### ⚙️*Module Description*

- Jailbreak results analysis by LongFormer (600M): `safety/longformer.py`, `safety/jailbreak_model_stat.py`, `safety/jailbreak_type_stat.py`
- Toxicity results analysis by Perspective API: `safety/perspective.py`, `misuse_stat.py`
- Misuse results analysis by LongFormer (600M): `safety/longformer.py`
- Exaggerated safety results analysis (manually)

#### 🔧*Usage*

```shell
python safety_eval.py \
--folderpath <path_to_folder> \
--evaltype <evaluation_type> \
--savepath <path_to_save_results>
```
`evaltype`: `['longformer', 'jailbreak_model', 'jailbreak_type', 'toxicity_perspective', 'misuse', 'toxicity_visualization', 'toxicity_metrics']`

- You must run `longformer` for all evaluation in safety section
- `jailbreak_model`: calculate the successfully jailbreak rate (1-RtA) for different LLMs.
- `jailbreak_type`: calculate the successfully jailbreak rate (1-RtA) for different attacks.
- `toxicity_perspective`: obtain toxicity value.
- `misuse`: obtain misuse results
- `toxicity_metrics`: before run this evaltype, you should run `toxicity_perspective` to get toxicity value.
- `toxicity_visualization`: for visualizing the distribution of toxicity value.

---

## Fairness


---

## Robustness

#### ⚙️*Module Description*

- AdvGLUE results analysis: `robustness/advglue_eval.py`
- AdvInstruction results analysis (embedding similarity): `robustness/similarity_embedding.py`

#### 🔧*Usage*

```shell
python robustness_eval.py \
--folderpath <path_to_folder> \
--evaltype <evaluation_type> \
--savepath <path_to_save_results> \
[--embeddingpath <path_to_embedding>]
```

`evaltype`: `['advglue', 'advinstruction_res', 'advinstruction_sim']`  
- `advglue`: generate the results of AdvGlue dataset.  
- `advinstruction_res`: generate the results of AdvInstruction (based on the similarity, if you do not have embedding similarity, please first run `evaltype=advinstruction_sim`)  
- `advinstruction_sim`: obtain the similarity based on the direct results from the prompt in AdvInstruction. (`embeddingpath` is necessary in this  `evaltype`)

---

## Privacy

#### ⚙️*Module Description*

- Privacy Awareness: `privacy/privacy_awareness.py`
- Privacy leakage: `privacy/privacy_leakage.py`

#### 🔧*Usage*
```shell
python privacy_eval.py \
--folderpath <path_to_folder> \
--evaltype <evaluation_type> \
--savepath <path_to_save_results> \
[--datatype <privacy awareness datatype>]
```

`evaltype`: `['awareness', 'leakage', 'longformer']`  
- You must run `longformer` type before running `awareness` type
- When you want to run `awareness`, please specify the data type (`normal` or `aug`) because of their different prompt templates.



---

## Ethics

#### ⚙️*Module Description*

- Implicit ethics: `ethics/social_norm.py` & `ethics/ETHICS.py`
- Explicit ethics: `ethics/jiminy.py`
- Emotional awareness: `ethics/emotinoal_awareness.py`

#### 🔧*Usage*

---