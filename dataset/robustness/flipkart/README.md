# Data description
Flipkart is a product review dataset released in 2022. Due to the unavailability of LLM's training data, we can consider these data as ‘out-of-distribution’ since they did not show up in, e.g., ChatGPT’s training data. 

This dataset can be used for **zero-shot classification** task. The flipkart.json file contain a reformat of the original dataset to form a test set. 

Format: 
```
{
        "prompt": "Is the following product review positive, neutral, or negative? Answer with \"positive\", \"neutral\", or \"negative\". Rating:5. Review: awesome. the cooler is really fantastic and provides good air flow highly recommended",
        "output": "positive"
}
```

# Flipkart review

Source: https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset?resource=download