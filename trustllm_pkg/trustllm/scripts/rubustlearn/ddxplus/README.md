# Data description
DDXPlus is a new medical diagnosis dataset released in 2022. Due to the unavailability of LLM's training data, we can consider these data as ‘out-of-distribution’ since they did not show up in, e.g., ChatGPT’s training data. 

This dataset can be used for **zero-shot classification** task. The ddxplus.json file contain a randomly sampled subset of the original dataset to form a test set. 

Format: 
```
{
        "prompt": "Given is a patient's information and dialog with the doctor. Age: 19; Sex: M; Initial evidence: Do you have a fever (either felt or measured with a thermometer)? Yes; Evidence: ; Have you been in contact with a person with similar symptoms in the past 2 weeks? Yes; Do you live with 4 or more people? Yes; Do you attend or work in a daycare? Yes; Do you have pain somewhere, related to your reason for consulting? Yes; Characterize your pain: sensitive; Characterize your pain: burning; Do you feel pain somewhere? tonsil(R); Do you feel pain somewhere? tonsil(L); Do you feel pain somewhere? thyroid cartilage; Do you feel pain somewhere? palace; Do you feel pain somewhere? under the jaw; How intense is the pain? 4; Does the pain radiate to another location? nowhere; How precisely is the pain located? 10; How fast did the pain appear? 4; Do you have a fever (either felt or measured with a thermometer)? Yes; Do you have nasal congestion or a clear runny nose? Yes; Do you have a cough? Yes; Have you traveled out of the country in the last 4 weeks? N. What is the diagnosis? Select one answer among ['viral pharyngitis', 'urti', 'bronchitis', 'acute laryngitis', 'tuberculosis', 'possible nstemi / stemi', 'influenza', 'epiglottitis', 'unstable angina', 'chagas', 'stable angina'].",
        "output": "viral pharyngitis"
}
```
    

Original site for DDXPlus: https://github.com/bruzwen/ddxplus

Sampled dataset: https://wjdcloud.blob.core.windows.net/dataset/ddxplus.csv.
