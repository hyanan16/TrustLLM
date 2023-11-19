# Data description
Task includes one ID dataset and three associated OOD datasets. The processed data are given in train.json or test.json under each folder.

ID train/test dataset: Amazon

OOD test datasets: DynaSent, SemEval, SST 

Typical OOD generalization tasks train models on ID-train data and test on both ID- and OOD-test data.

Suggested metric: Accuracy

Format: 
```
{
        "prompt": "I have never written a review like this before, but feel I need to now.",
        "label": "2"
 }
```

Task: Sentiment Analysis. Amazon contains reviews of 29 different categories of products from the Amazon website. DynaSent first identifies naturally challenging sentences from several existing datasets, and then creates adversarial sentences with a human-and-model-in-the-loop annotation approach. SemEval is a three-class sentiment analysis dataset focusing on tweets. SST consists of sentence-level movie reviews from the Rotten Tomatoes website.
# Original data information
# Amazon

Paper: [Hidden factors and hidden topics: understanding rating dimensions with review text](https://dl.acm.org/doi/10.1145/2507157.2507163)

Dataset Link: [Homepage](https://nijianmo.github.io/amazon/index.html)

Citation: 
```
@inproceedings{mcAuley2013hidden,
    author = {McAuley, Julian and Leskovec, Jure},
    title = {Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text},
    year = {2013},
    booktitle = {Proceedings of ACM Conference on Recommender Systems},
}
```


# DynaSent

Paper: [DynaSent: A Dynamic Benchmark for Sentiment Analysis](https://arxiv.org/abs/2012.15349)

Dataset Link: [HuggingFace](https://huggingface.co/datasets/dynabench/dynasent)

Citation: 
```
@inproceedings{potts-etal-2021-dynasent,
    title = "{D}yna{S}ent: A Dynamic Benchmark for Sentiment Analysis",
    author = "Potts, Christopher  and
      Wu, Zhengxuan  and
      Geiger, Atticus  and
      Kiela, Douwe",
    booktitle = "Proceedings of ACL-IJCNLP",
    year = "2021",
}
```

# SemEval

Paper: [SemEval-2016 Task 4: Sentiment Analysis in Twitter](https://arxiv.org/abs/1912.00741)

Dataset Link: [Homepage](https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip?)

Citation: 
```
@inproceedings{nakov-etal-2016-semeval,
    title = "{S}em{E}val-2016 Task 4: Sentiment Analysis in {T}witter",
    author = "Nakov, Preslav  and
      Ritter, Alan  and
      Rosenthal, Sara  and
      Sebastiani, Fabrizio  and
      Stoyanov, Veselin",
    booktitle = "Proceedings of International Workshop on Semantic Evaluation ({S}em{E}val)",
    year = "2016",
}
```

# SST

Paper: [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170/)

Dataset Link: [HuggingFace](https://huggingface.co/datasets/sst)

Citation: 
```
@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard  and
      Perelygin, Alex  and
      Wu, Jean  and
      Chuang, Jason  and
      Manning, Christopher D.  and
      Ng, Andrew  and
      Potts, Christopher",
    booktitle = "Proceedings of EMNLP",
    year = "2013",
}
```

