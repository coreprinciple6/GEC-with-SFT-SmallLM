## TASK
Perform grammatical error correction (GEC) task using the Grammarly CoEdIT dataset. The data is in json format. here is a sample from the dataset

```
{
  '_id': 1,
  'task': "gec",
  'src': "Improve the grammaticality: As the number of people grows, the need of habitable environment is unquestionably essential.",
  'tgt': "As the number of people grows, the need for a habitable environment is unquestionably increasing."
}
```

The task is to  Fine-tune the [SmolLM-135M model](https://huggingface.co/HuggingFaceTB/SmolLM-135M) using the CoEdIT datasetwhich includes input sentences with grammatical errors and their corrected versions. Use the training GEC portion of the CoEdIT dataset to teach the model how to correct grammatical errors effectively. Calculate the BLEU score on the validation set to evaluate the model's performance in generating grammatically correct sentences.
