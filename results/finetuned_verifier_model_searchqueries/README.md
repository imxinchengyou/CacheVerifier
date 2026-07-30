---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:60398
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['best way to cook turkey legs', 'Not required for the benchmark because of the id_set'],
    ['best way to burn belly fat', 'Not required for the benchmark because of the id_set'],
    ['crockpot pork tenderloin slow cooker recipes', 'Not required for the benchmark because of the id_set'],
    ['bed bath and beyond schaumburg', 'Not required for the benchmark because of the id_set'],
    ['brother printer download for windows 10', 'Not required for the benchmark because of the id_set'],
]
scores = model.predict(pairs)
print(scores)
# [ 0.3895  0.9268 -0.1011 -0.6965  0.1626]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'best way to cook turkey legs',
    [
        'Not required for the benchmark because of the id_set',
        'Not required for the benchmark because of the id_set',
        'Not required for the benchmark because of the id_set',
        'Not required for the benchmark because of the id_set',
        'Not required for the benchmark because of the id_set',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 60,398 training samples
* Columns: <code>query</code>, <code>response</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | query                                                                            | response                                                                          | label                                                          |
  |:---------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                           | string                                                                            | float                                                          |
  | modality | text                                                                             | text                                                                              |                                                                |
  | details  | <ul><li>min: 7 tokens</li><li>mean: 8.26 tokens</li><li>max: 13 tokens</li></ul> | <ul><li>min: 14 tokens</li><li>mean: 14.0 tokens</li><li>max: 14 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.19</li><li>max: 1.0</li></ul> |
* Samples:
  | query                                                     | response                                                          | label            |
  |:----------------------------------------------------------|:------------------------------------------------------------------|:-----------------|
  | <code>best way to cook turkey legs</code>                 | <code>Not required for the benchmark because of the id_set</code> | <code>0.0</code> |
  | <code>best way to burn belly fat</code>                   | <code>Not required for the benchmark because of the id_set</code> | <code>0.0</code> |
  | <code>crockpot pork tenderloin slow cooker recipes</code> | <code>Not required for the benchmark because of the id_set</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `disable_tqdm`: True
- `use_cpu`: True
- `dataloader_pin_memory`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: True
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: True
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: False
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0003 | 1    | 6.3239        |
| 0.0026 | 10   | 2.3613        |
| 0.0053 | 20   | 0.8354        |
| 0.0079 | 30   | 0.7806        |
| 0.0106 | 40   | 0.7094        |
| 0.0132 | 50   | 0.6971        |
| 0.0159 | 60   | 0.7426        |
| 0.0185 | 70   | 0.7030        |
| 0.0212 | 80   | 0.7072        |
| 0.0238 | 90   | 0.7494        |
| 0.0265 | 100  | 0.7859        |
| 0.0291 | 110  | 0.7176        |
| 0.0318 | 120  | 0.7114        |
| 0.0344 | 130  | 0.6915        |
| 0.0371 | 140  | 0.7053        |
| 0.0397 | 150  | 0.7563        |
| 0.0424 | 160  | 0.6967        |
| 0.0450 | 170  | 0.6692        |
| 0.0477 | 180  | 0.6960        |
| 0.0503 | 190  | 0.6936        |
| 0.0530 | 200  | 0.7033        |
| 0.0556 | 210  | 0.6745        |
| 0.0583 | 220  | 0.6698        |
| 0.0609 | 230  | 0.6556        |
| 0.0636 | 240  | 0.7094        |
| 0.0662 | 250  | 0.6900        |
| 0.0689 | 260  | 0.7115        |
| 0.0715 | 270  | 0.7050        |
| 0.0742 | 280  | 0.6753        |
| 0.0768 | 290  | 0.6966        |
| 0.0795 | 300  | 0.6800        |
| 0.0821 | 310  | 0.6921        |
| 0.0848 | 320  | 0.6933        |
| 0.0874 | 330  | 0.6817        |
| 0.0901 | 340  | 0.6741        |
| 0.0927 | 350  | 0.6949        |
| 0.0954 | 360  | 0.6899        |
| 0.0980 | 370  | 0.6621        |
| 0.1007 | 380  | 0.6972        |
| 0.1033 | 390  | 0.6563        |
| 0.1060 | 400  | 0.6914        |
| 0.1086 | 410  | 0.6653        |
| 0.1113 | 420  | 0.6908        |
| 0.1139 | 430  | 0.6394        |
| 0.1166 | 440  | 0.6460        |
| 0.1192 | 450  | 0.7372        |
| 0.1219 | 460  | 0.6722        |
| 0.1245 | 470  | 0.6820        |
| 0.1272 | 480  | 0.6747        |
| 0.1298 | 490  | 0.6916        |
| 0.1325 | 500  | 0.6960        |
| 0.1351 | 510  | 0.7122        |
| 0.1377 | 520  | 0.7192        |
| 0.1404 | 530  | 0.7004        |
| 0.1430 | 540  | 0.6781        |
| 0.1457 | 550  | 0.6751        |
| 0.1483 | 560  | 0.6711        |
| 0.1510 | 570  | 0.6668        |
| 0.1536 | 580  | 0.6739        |
| 0.1563 | 590  | 0.6725        |
| 0.1589 | 600  | 0.6646        |
| 0.1616 | 610  | 0.6831        |
| 0.1642 | 620  | 0.6947        |
| 0.1669 | 630  | 0.6557        |
| 0.1695 | 640  | 0.6711        |
| 0.1722 | 650  | 0.6855        |
| 0.1748 | 660  | 0.6830        |
| 0.1775 | 670  | 0.7071        |
| 0.1801 | 680  | 0.6618        |
| 0.1828 | 690  | 0.6639        |
| 0.1854 | 700  | 0.6806        |
| 0.1881 | 710  | 0.6567        |
| 0.1907 | 720  | 0.6841        |
| 0.1934 | 730  | 0.7029        |
| 0.1960 | 740  | 0.6743        |
| 0.1987 | 750  | 0.6793        |
| 0.2013 | 760  | 0.6827        |
| 0.2040 | 770  | 0.6824        |
| 0.2066 | 780  | 0.6897        |
| 0.2093 | 790  | 0.6659        |
| 0.2119 | 800  | 0.6537        |
| 0.2146 | 810  | 0.7122        |
| 0.2172 | 820  | 0.6822        |
| 0.2199 | 830  | 0.6784        |
| 0.2225 | 840  | 0.6561        |
| 0.2252 | 850  | 0.6743        |
| 0.2278 | 860  | 0.6696        |
| 0.2305 | 870  | 0.7039        |
| 0.2331 | 880  | 0.6510        |
| 0.2358 | 890  | 0.6867        |
| 0.2384 | 900  | 0.6511        |
| 0.2411 | 910  | 0.6679        |
| 0.2437 | 920  | 0.6450        |
| 0.2464 | 930  | 0.6908        |
| 0.2490 | 940  | 0.6500        |
| 0.2517 | 950  | 0.6531        |
| 0.2543 | 960  | 0.6798        |
| 0.2570 | 970  | 0.6809        |
| 0.2596 | 980  | 0.6596        |
| 0.2623 | 990  | 0.6696        |
| 0.2649 | 1000 | 0.6821        |
| 0.2675 | 1010 | 0.6982        |
| 0.2702 | 1020 | 0.6839        |
| 0.2728 | 1030 | 0.6701        |
| 0.2755 | 1040 | 0.6734        |
| 0.2781 | 1050 | 0.6749        |
| 0.2808 | 1060 | 0.6709        |
| 0.2834 | 1070 | 0.6661        |
| 0.2861 | 1080 | 0.6681        |
| 0.2887 | 1090 | 0.6792        |
| 0.2914 | 1100 | 0.6560        |
| 0.2940 | 1110 | 0.6694        |
| 0.2967 | 1120 | 0.6811        |
| 0.2993 | 1130 | 0.6631        |
| 0.3020 | 1140 | 0.6894        |
| 0.3046 | 1150 | 0.6542        |
| 0.3073 | 1160 | 0.6929        |
| 0.3099 | 1170 | 0.6509        |
| 0.3126 | 1180 | 0.6727        |
| 0.3152 | 1190 | 0.6511        |
| 0.3179 | 1200 | 0.7209        |
| 0.3205 | 1210 | 0.6500        |
| 0.3232 | 1220 | 0.6674        |
| 0.3258 | 1230 | 0.6634        |
| 0.3285 | 1240 | 0.6628        |
| 0.3311 | 1250 | 0.6937        |
| 0.3338 | 1260 | 0.7146        |
| 0.3364 | 1270 | 0.6647        |
| 0.3391 | 1280 | 0.6644        |
| 0.3417 | 1290 | 0.6712        |
| 0.3444 | 1300 | 0.6604        |
| 0.3470 | 1310 | 0.6856        |
| 0.3497 | 1320 | 0.6534        |
| 0.3523 | 1330 | 0.6813        |
| 0.3550 | 1340 | 0.6401        |
| 0.3576 | 1350 | 0.6840        |
| 0.3603 | 1360 | 0.6897        |
| 0.3629 | 1370 | 0.6462        |
| 0.3656 | 1380 | 0.6392        |
| 0.3682 | 1390 | 0.6651        |
| 0.3709 | 1400 | 0.6800        |
| 0.3735 | 1410 | 0.6613        |
| 0.3762 | 1420 | 0.6641        |
| 0.3788 | 1430 | 0.6713        |
| 0.3815 | 1440 | 0.6552        |
| 0.3841 | 1450 | 0.6705        |
| 0.3868 | 1460 | 0.6742        |
| 0.3894 | 1470 | 0.6678        |
| 0.3921 | 1480 | 0.6824        |
| 0.3947 | 1490 | 0.6651        |
| 0.3974 | 1500 | 0.6842        |
| 0.4    | 1510 | 0.7100        |
| 0.4026 | 1520 | 0.6701        |
| 0.4053 | 1530 | 0.6723        |
| 0.4079 | 1540 | 0.6745        |
| 0.4106 | 1550 | 0.6580        |
| 0.4132 | 1560 | 0.6322        |
| 0.4159 | 1570 | 0.6347        |
| 0.4185 | 1580 | 0.6652        |
| 0.4212 | 1590 | 0.7046        |
| 0.4238 | 1600 | 0.6408        |
| 0.4265 | 1610 | 0.6674        |
| 0.4291 | 1620 | 0.6636        |
| 0.4318 | 1630 | 0.6816        |
| 0.4344 | 1640 | 0.6868        |
| 0.4371 | 1650 | 0.6697        |
| 0.4397 | 1660 | 0.6501        |
| 0.4424 | 1670 | 0.6537        |
| 0.4450 | 1680 | 0.7102        |
| 0.4477 | 1690 | 0.6490        |
| 0.4503 | 1700 | 0.6599        |
| 0.4530 | 1710 | 0.6859        |
| 0.4556 | 1720 | 0.6616        |
| 0.4583 | 1730 | 0.6552        |
| 0.4609 | 1740 | 0.6914        |
| 0.4636 | 1750 | 0.6773        |
| 0.4662 | 1760 | 0.6324        |
| 0.4689 | 1770 | 0.6571        |
| 0.4715 | 1780 | 0.6784        |
| 0.4742 | 1790 | 0.6569        |
| 0.4768 | 1800 | 0.6735        |
| 0.4795 | 1810 | 0.6681        |
| 0.4821 | 1820 | 0.6513        |
| 0.4848 | 1830 | 0.6602        |
| 0.4874 | 1840 | 0.6669        |
| 0.4901 | 1850 | 0.6948        |
| 0.4927 | 1860 | 0.6695        |
| 0.4954 | 1870 | 0.6621        |
| 0.4980 | 1880 | 0.6761        |
| 0.5007 | 1890 | 0.6783        |
| 0.5033 | 1900 | 0.6700        |
| 0.5060 | 1910 | 0.6388        |
| 0.5086 | 1920 | 0.6365        |
| 0.5113 | 1930 | 0.6893        |
| 0.5139 | 1940 | 0.6545        |
| 0.5166 | 1950 | 0.6629        |
| 0.5192 | 1960 | 0.6736        |
| 0.5219 | 1970 | 0.6439        |
| 0.5245 | 1980 | 0.6880        |
| 0.5272 | 1990 | 0.6516        |
| 0.5298 | 2000 | 0.6213        |
| 0.5325 | 2010 | 0.6618        |
| 0.5351 | 2020 | 0.6710        |
| 0.5377 | 2030 | 0.6811        |
| 0.5404 | 2040 | 0.6554        |
| 0.5430 | 2050 | 0.6808        |
| 0.5457 | 2060 | 0.6454        |
| 0.5483 | 2070 | 0.6469        |
| 0.5510 | 2080 | 0.6597        |
| 0.5536 | 2090 | 0.7051        |
| 0.5563 | 2100 | 0.6778        |
| 0.5589 | 2110 | 0.6403        |
| 0.5616 | 2120 | 0.6389        |
| 0.5642 | 2130 | 0.6543        |
| 0.5669 | 2140 | 0.6175        |
| 0.5695 | 2150 | 0.6715        |
| 0.5722 | 2160 | 0.6270        |
| 0.5748 | 2170 | 0.6429        |
| 0.5775 | 2180 | 0.6719        |
| 0.5801 | 2190 | 0.6637        |
| 0.5828 | 2200 | 0.7192        |
| 0.5854 | 2210 | 0.6434        |
| 0.5881 | 2220 | 0.6628        |
| 0.5907 | 2230 | 0.6289        |
| 0.5934 | 2240 | 0.6471        |
| 0.5960 | 2250 | 0.6860        |
| 0.5987 | 2260 | 0.6670        |
| 0.6013 | 2270 | 0.6372        |
| 0.6040 | 2280 | 0.6617        |
| 0.6066 | 2290 | 0.6579        |
| 0.6093 | 2300 | 0.6860        |
| 0.6119 | 2310 | 0.6609        |
| 0.6146 | 2320 | 0.6532        |
| 0.6172 | 2330 | 0.6360        |
| 0.6199 | 2340 | 0.6392        |
| 0.6225 | 2350 | 0.6771        |
| 0.6252 | 2360 | 0.6572        |
| 0.6278 | 2370 | 0.6558        |
| 0.6305 | 2380 | 0.6665        |
| 0.6331 | 2390 | 0.6455        |
| 0.6358 | 2400 | 0.6686        |
| 0.6384 | 2410 | 0.6423        |
| 0.6411 | 2420 | 0.6620        |
| 0.6437 | 2430 | 0.6718        |
| 0.6464 | 2440 | 0.6811        |
| 0.6490 | 2450 | 0.6204        |
| 0.6517 | 2460 | 0.6433        |
| 0.6543 | 2470 | 0.6468        |
| 0.6570 | 2480 | 0.6544        |
| 0.6596 | 2490 | 0.6720        |
| 0.6623 | 2500 | 0.6504        |
| 0.6649 | 2510 | 0.6707        |
| 0.6675 | 2520 | 0.6312        |
| 0.6702 | 2530 | 0.6743        |
| 0.6728 | 2540 | 0.6454        |
| 0.6755 | 2550 | 0.6605        |
| 0.6781 | 2560 | 0.6966        |
| 0.6808 | 2570 | 0.6981        |
| 0.6834 | 2580 | 0.6538        |
| 0.6861 | 2590 | 0.6305        |
| 0.6887 | 2600 | 0.6261        |
| 0.6914 | 2610 | 0.6699        |
| 0.6940 | 2620 | 0.6928        |
| 0.6967 | 2630 | 0.6835        |
| 0.6993 | 2640 | 0.6472        |
| 0.7020 | 2650 | 0.6489        |
| 0.7046 | 2660 | 0.6324        |
| 0.7073 | 2670 | 0.6651        |
| 0.7099 | 2680 | 0.6197        |
| 0.7126 | 2690 | 0.6653        |
| 0.7152 | 2700 | 0.6447        |
| 0.7179 | 2710 | 0.6584        |
| 0.7205 | 2720 | 0.6244        |
| 0.7232 | 2730 | 0.6692        |
| 0.7258 | 2740 | 0.6655        |
| 0.7285 | 2750 | 0.6759        |
| 0.7311 | 2760 | 0.6265        |
| 0.7338 | 2770 | 0.6118        |
| 0.7364 | 2780 | 0.6519        |
| 0.7391 | 2790 | 0.6521        |
| 0.7417 | 2800 | 0.6787        |
| 0.7444 | 2810 | 0.6696        |
| 0.7470 | 2820 | 0.6603        |
| 0.7497 | 2830 | 0.6612        |
| 0.7523 | 2840 | 0.6922        |
| 0.7550 | 2850 | 0.6549        |
| 0.7576 | 2860 | 0.6559        |
| 0.7603 | 2870 | 0.6887        |
| 0.7629 | 2880 | 0.6341        |
| 0.7656 | 2890 | 0.6654        |
| 0.7682 | 2900 | 0.6560        |
| 0.7709 | 2910 | 0.6697        |
| 0.7735 | 2920 | 0.6619        |
| 0.7762 | 2930 | 0.6788        |
| 0.7788 | 2940 | 0.6677        |
| 0.7815 | 2950 | 0.6540        |
| 0.7841 | 2960 | 0.6338        |
| 0.7868 | 2970 | 0.6652        |
| 0.7894 | 2980 | 0.6412        |
| 0.7921 | 2990 | 0.6196        |
| 0.7947 | 3000 | 0.6460        |
| 0.7974 | 3010 | 0.6813        |
| 0.8    | 3020 | 0.6419        |
| 0.8026 | 3030 | 0.6337        |
| 0.8053 | 3040 | 0.6388        |
| 0.8079 | 3050 | 0.6527        |
| 0.8106 | 3060 | 0.6488        |
| 0.8132 | 3070 | 0.6254        |
| 0.8159 | 3080 | 0.6511        |
| 0.8185 | 3090 | 0.6796        |
| 0.8212 | 3100 | 0.7048        |
| 0.8238 | 3110 | 0.6296        |
| 0.8265 | 3120 | 0.6707        |
| 0.8291 | 3130 | 0.6483        |
| 0.8318 | 3140 | 0.6587        |
| 0.8344 | 3150 | 0.6837        |
| 0.8371 | 3160 | 0.6621        |
| 0.8397 | 3170 | 0.6236        |
| 0.8424 | 3180 | 0.6413        |
| 0.8450 | 3190 | 0.6657        |
| 0.8477 | 3200 | 0.6531        |
| 0.8503 | 3210 | 0.6318        |
| 0.8530 | 3220 | 0.6589        |
| 0.8556 | 3230 | 0.6782        |
| 0.8583 | 3240 | 0.6370        |
| 0.8609 | 3250 | 0.6414        |
| 0.8636 | 3260 | 0.7087        |
| 0.8662 | 3270 | 0.6476        |
| 0.8689 | 3280 | 0.6675        |
| 0.8715 | 3290 | 0.6981        |
| 0.8742 | 3300 | 0.6641        |
| 0.8768 | 3310 | 0.6864        |
| 0.8795 | 3320 | 0.6552        |
| 0.8821 | 3330 | 0.6489        |
| 0.8848 | 3340 | 0.6577        |
| 0.8874 | 3350 | 0.6529        |
| 0.8901 | 3360 | 0.6604        |
| 0.8927 | 3370 | 0.6464        |
| 0.8954 | 3380 | 0.6648        |
| 0.8980 | 3390 | 0.6582        |
| 0.9007 | 3400 | 0.6512        |
| 0.9033 | 3410 | 0.6796        |
| 0.9060 | 3420 | 0.6695        |
| 0.9086 | 3430 | 0.6637        |
| 0.9113 | 3440 | 0.6601        |
| 0.9139 | 3450 | 0.6593        |
| 0.9166 | 3460 | 0.6505        |
| 0.9192 | 3470 | 0.6532        |
| 0.9219 | 3480 | 0.6825        |
| 0.9245 | 3490 | 0.6737        |
| 0.9272 | 3500 | 0.6452        |
| 0.9298 | 3510 | 0.6332        |
| 0.9325 | 3520 | 0.6438        |
| 0.9351 | 3530 | 0.6038        |
| 0.9377 | 3540 | 0.6623        |
| 0.9404 | 3550 | 0.6432        |
| 0.9430 | 3560 | 0.6714        |
| 0.9457 | 3570 | 0.6532        |
| 0.9483 | 3580 | 0.6843        |
| 0.9510 | 3590 | 0.6711        |
| 0.9536 | 3600 | 0.6585        |
| 0.9563 | 3610 | 0.6231        |
| 0.9589 | 3620 | 0.6513        |
| 0.9616 | 3630 | 0.6655        |
| 0.9642 | 3640 | 0.6641        |
| 0.9669 | 3650 | 0.6610        |
| 0.9695 | 3660 | 0.6343        |
| 0.9722 | 3670 | 0.6740        |
| 0.9748 | 3680 | 0.6601        |
| 0.9775 | 3690 | 0.6494        |
| 0.9801 | 3700 | 0.6676        |
| 0.9828 | 3710 | 0.6647        |
| 0.9854 | 3720 | 0.6475        |
| 0.9881 | 3730 | 0.6464        |
| 0.9907 | 3740 | 0.6686        |
| 0.9934 | 3750 | 0.6371        |
| 0.9960 | 3760 | 0.6968        |
| 0.9987 | 3770 | 0.6851        |

</details>

### Training Time
- **Training**: 5.7 minutes

### Framework Versions
- Python: 3.11.6
- Sentence Transformers: 5.6.1
- Transformers: 5.14.1
- PyTorch: 2.13.0+cu130
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->