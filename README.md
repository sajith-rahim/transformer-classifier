


*A Transformer Classifier implemented from Scratch.*

<p>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" />

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" />
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Shell_Script-121011?style=for-the-badge&logo=gnu-bash&logoColor=white" />
</p>

### Paper

<a href="https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf" target="_blank">Attention Is All You Need. Ashish Vaswani, et al. NIPS 2017</a>.

## Getting Started



### Prerequisites

| Package     | Version      |
|:----------------|:---------------|
| torch| 1.10.0 |
| torchvision| 0.11.1 |
| omegaconf| 2.1.1 |
| validators| 1.18.5 |
| matplotlib|3.4.1 |
| requests|2.22.0 |
| hydra_core| 1.1.1 |
| dataclasses| 0.6 |
| numpy| 1.18.5 |
| tqdm| 4.62.3 |
| nltk| 3.7 |

Note: Additional requirements are from *papyrus*.
```powershell
─────────────╔═╗
╔═╦═╗╔═╦╦╦╦╦╦╣═╣
║╬║╬╚╣╬║║║╔╣║╠═║
║╔╩══╣╔╬╗╠╝╚═╩═╝
╚╝───╚╝╚═╝
```

### Installing

```powershell
pip install -r requirements.txt
```

### Inference
```powershell
python infer.py
```
#### Output
```powershell
FC Barcelona finally accepts it has turned into shit.
Category: Sports, Probability: 96.03%
```



### Download scripts


Example: AG's News Topic Classification Dataset

The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.

Origin:  <a href="http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html" target="_blank">AG Corpus</a>.

```powershell
Classes:
  >  World
  >  Sports
  >  Business
  >  Sci/Tech
```
```powershell
wget https://data.deepai.org/agnews.zip
```
or

Use downloader:
```powershell
downloader = FileDownloader()
downloader.download_file(url, filename=None, target_dir="./data/raw")
```

## Training 

From the root directory run

```powershell
python main.py
```


#### Output

```powershell
files:
  test_data: test_data.pth.tar
  test_labels: None
  train_data: train_data.pth.tar
  train_labels: None
paths:
  log: /content/tformer-clf/logs
  data: /content/tformer-clf/data
params:
  epoch_count: 200
  lr: 5.0e-05
  batch_size: 128
  shuffle: true
  num_workers: 2
  query_limit: 200
  emb_size: 256
  n_heads: 8
  n_encoders: 2
  ffn_hidden_size: 512
  classifier_hidden_layer_size: 2048
  dropout: 0.3
checkpoint:
  save_interval: 10
  resume: false
  checkpoint_id: TransformerClassifier-10_02_2022_23_43_34-20-0.95.pt.zip
  path: /content/tformer-clf/checkpoints

Active device: cuda
TransformerClassifier(
  (embeddings): Embedding(31994, 256)
  (postional_encoding): PositionalEncoding(
    (dropout): Dropout(p=0.3, inplace=False)
  )
  (encoder): TransformerEncoderBlock(
    (attn): MultiHeadScaledDotProductAttention(
      (W_q): Linear(in_features=256, out_features=256, bias=True)
      (W_k): Linear(in_features=256, out_features=256, bias=True)
      (W_v): Linear(in_features=256, out_features=256, bias=True)
      (dropout): Dropout(p=0.3, inplace=False)
      (softmax): Softmax(dim=-1)
      (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (fc): Linear(in_features=256, out_features=256, bias=True)
      (dropout2): Dropout(p=0.3, inplace=False)
    )
    (feed_forward): PositionWiseFFN(
      (W_1): Linear(in_features=256, out_features=512, bias=True)
      (W_2): Linear(in_features=512, out_features=256, bias=True)
      (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.3, inplace=False)
      (relu): ReLU()
    )
  )
  (encoders): ModuleList(
    (0): TransformerEncoderBlock(
      (attn): MultiHeadScaledDotProductAttention(
        (W_q): Linear(in_features=256, out_features=256, bias=True)
        (W_k): Linear(in_features=256, out_features=256, bias=True)
        (W_v): Linear(in_features=256, out_features=256, bias=True)
        (dropout): Dropout(p=0.3, inplace=False)
        (softmax): Softmax(dim=-1)
        (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (fc): Linear(in_features=256, out_features=256, bias=True)
        (dropout2): Dropout(p=0.3, inplace=False)
      )
      (feed_forward): PositionWiseFFN(
        (W_1): Linear(in_features=256, out_features=512, bias=True)
        (W_2): Linear(in_features=512, out_features=256, bias=True)
        (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.3, inplace=False)
        (relu): ReLU()
      )
    )
    (1): TransformerEncoderBlock(
      (attn): MultiHeadScaledDotProductAttention(
        (W_q): Linear(in_features=256, out_features=256, bias=True)
        (W_k): Linear(in_features=256, out_features=256, bias=True)
        (W_v): Linear(in_features=256, out_features=256, bias=True)
        (dropout): Dropout(p=0.3, inplace=False)
        (softmax): Softmax(dim=-1)
        (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (fc): Linear(in_features=256, out_features=256, bias=True)
        (dropout2): Dropout(p=0.3, inplace=False)
      )
      (feed_forward): PositionWiseFFN(
        (W_1): Linear(in_features=256, out_features=512, bias=True)
        (W_2): Linear(in_features=512, out_features=256, bias=True)
        (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.3, inplace=False)
        (relu): ReLU()
      )
    )
  )
  (fc): Linear(in_features=51200, out_features=2048, bias=True)
  (fc2): Linear(in_features=2048, out_features=4, bias=True)
)
Trainable parameters: 114639620

Train Progress:   5%|███                                                               | 44/938 [00:19<06:39,  2.24it/s]
```

To override params from CLI:

```powershell
python main.py params.num_workers=2
```

To override params from CLI if exists else add:

```powershell
python main.py ++params.other_params=abc
```

## Checkpointing

Set the resume flag and checkpoint name in `config.yaml`

```yaml
*config.yaml*
------------
defaults:
  ...
paths:
  ...
params:
  ...
checkpoint:
  save_interval: 5 #iter interval for checkpointing
  resume: False # resume flag if True set checkpoint_id
  checkpoint_id: <ID>.pt.zip # saved checkpoint filename
  path: ${hydra:runtime.cwd}/checkpoints #save path
```

## Tensorboard Server

Check the name of your log folder under `config.defaults.log`

```yaml
*config.yaml*
------------
defaults:
  ...
paths:
  log: ${hydra:runtime.cwd}/runs
  data: ...
params:
  ...
```


From project root run
```powershell
tensorboard --logdir runs
```

tensorboard server will open at

    http://localhost:6006

# Folder Structure

```powershell
|   down_mnist.sh
|   infer.py
|   main.py
|   note.txt
|   README.md
|   requirements.txt
|
+---base
|       base_dataloader.py
|       base_dataset.py
|       base_model.py
|       __init__.py
|
+---config
|   |   clf_config.py
|   |   config.py
|   |   __init__.py
|   |
|   \---conf
|       |   config.yaml
|       |
|       \---files
|               agnews.yaml
|
+---data
|       test_data.pth.tar
|       train_data.pth.tar
|       word_map.json
|
+---dataloader
|   |   sentence_dataloader.py
|   |   __init__.py
|   |
|   \---dataset
|           sentence_dataset.py
|           __init__.py
|
+---logs
|   +---0
|   |       events.out.tfevents.1644523642.Drovahkin.12564.0
|   
+---metrics
|       losses.py
|       metric.py
|       __init__.py
|
+---models
|   |   transformer.py
|   |   __init__.py
|   |
|   \---blocks
|           attention.py
|           encoder_block.py
|           positional_encoding.py
|           pos_wise_ffn.py
|           __init__.py
|
+---outputs
+---store
|       checkpointer.py
|       runtime_kvstore.py
|       __init__.py
|
+---task_runner
|       task_runner.py
|       __init__.py
|
+---tracker
|       phase.py
|       tensorboard_experiment.py
|       track.py
|       __init__.py
|
\---utils
        config_utils.py
        data_utils.py
        device_utils.py
        download_utils.py
        os_utils.py
        tracker_utils.py
        __init__.py
```
## License

    MIT

## Future

<img align="right" style="float:right;border:3px solid black" width=64 height=92 src="https://raw.githubusercontent.com/sajith-rahim/cdn/main/content/blog/media/warn_tag.png" />

 * Support pre trained word embeddings.

## Acknowledgments

Tensor2Tensor <a href="https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py" target="_blank">transformer.py</a>
