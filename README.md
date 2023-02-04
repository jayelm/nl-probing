# nl-probing

Probing for natural language explanations.

Configure model and dataset to probe in `src/conf/generate_hidden_config.yaml`. Default is [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) on [esnli](https://huggingface.co/datasets/esnli) (which is just SNLI). Then run

```
python -m src.generate_hidden
```

to generate an `.hdf5` file with hidden states of the model in `hidden_states/`.
