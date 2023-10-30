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


---

## Fairness


---

## Robustness

#### ⚙️*Module Description*

- AdvGLUE results analysis: `robustness/advglue_eval.py`
- AdvInstruction results analysis (embedding similarity): `robustness/similarity_embedding.py`

#### 🔧*Usage*

```shell
python robustness.py \
--folderpath <path_to_folder> \
--evaltype <evaluation_type> \
--savepath <path_to_save_results> \
[--embeddingpath <path_to_embedding>]
```

`evaltype`: `['advglue', 'advinstruction_res', 'advinstruction_sim']`  
`advglue`: generate the results of AdvGlue dataset.  
`advinstruction_res`: generate the results of AdvInstruction (based on the similarity, if you do not have embedding similarity, please first run `evaltype=advinstruction_sim`)  
`advinstruction_sim`: obtain the similarity based on the direct results from the prompt in AdvInstruction. (`embeddingpath` is necessary in this  `evaltype`)

---

## Privacy

#### ⚙️*Module Description*

- Privacy leakage: `privacy/privacy_leakage.py`

#### 🔧*Usage*

---

## Ethics

#### ⚙️*Module Description*

- Implicit ethics: `ethics/social_norm.py` & `ethics/ETHICS.py`
- Explicit ethics: `ethics/jiminy.py`

#### 🔧*Usage*

---