import argparse
import sys

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import ClfConfig
from dataloader.sentence_dataloader import SentenceDataLoader, build_vocab
from models import TransformerClassifier

from store import Checkpointer
from task_runner.task_runner import TaskRunner
from tracker import TensorboardExperiment, Phase
from utils import merge_config, get_device

# 1. initialize config store
cs = ConfigStore.instance()

# 2. load project config schema
cs.store(name="clf_config", node=ClfConfig)


# 3. set path and filename
@hydra.main(config_path="config/conf", config_name="config")
def run(config: ClfConfig) -> None:
    # overwrite with cli params
    print(OmegaConf.from_cli())
    config = merge_config(config, OmegaConf.from_cli())

    OmegaConf.set_readonly(config, True)
    print(OmegaConf.to_yaml(config, resolve=True))
    print(f"Active device: {get_device()}")

    # 4. define data-loaders

    data_loader = SentenceDataLoader(
        config.params.batch_size,
        config.params.shuffle,
        config.params.num_workers
    )

    test_loader = data_loader.create_dataloader(
        root_path=config.paths.data,
        data_file=config.files.test_data,
        label_file=config.files.test_labels,
    )
    train_loader = data_loader.create_dataloader(
        root_path=config.paths.data,
        data_file=config.files.train_data,
        label_file=config.files.train_labels,
    )

    embeddings, emb_size, word_map, n_classes, vocab_size = build_vocab(config.paths.data, config.params.emb_size,
                                                                        False)

    # 5. define model
    model_kwargs = {
        'dim': emb_size,
        'q_len': config.params.query_limit,
        'ffn_hidden_layer_size': config.params.ffn_hidden_size,
        'n_heads': config.params.n_heads,
        'n_encoder': config.params.n_encoders,
        'classfifier_hidden_layer_size': config.params.classifier_hidden_layer_size,
        'dropout': config.params.dropout
    }
    model = TransformerClassifier(n_classes, vocab_size, embeddings, **model_kwargs).to(get_device())
    print(model.__str__())

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.params.lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    # TODO - Refactor
    # if resume; load checkpoints before initializing task runners
    # checkpoint = Checkpointer.load_checkpoint(config.checkpoint.checkpoint_id, config.checkpoint.path,
    #                                         str(get_device()))

    # 6. define task runners
    test_runner = TaskRunner(Phase.VAL, test_loader, model, loss_fn, config.checkpoint)
    train_runner = TaskRunner(Phase.TRAIN, train_loader, model, loss_fn, config.checkpoint, optimizer)

    # 7. define tracker and set log dir
    tracker = TensorboardExperiment(log_path=config.paths.log)

    # tracker.add_graph(model, iter(test_runner.dataloader).next()[0])

    # 8. train
    for epoch_id in range(config.params.epoch_count):

        # resume
        if config.checkpoint.resume:
            checkpoint_cfg = config.checkpoint
            # load state dict
            tr_iter = train_runner.load_checkpoint(checkpoint_cfg)
            ts_iter = test_runner.load_checkpoint(checkpoint_cfg)
            if ts_iter == -1 or tr_iter == -1:
                print("Checkpoint load failed.")
            assert ts_iter == tr_iter, "Inconsistency in checkpoint."
            epoch_id = tr_iter + 1

            print(f"{config.checkpoint.checkpoint_id} : Resuming from {epoch_id} of {config.params.epoch_count}")

        TaskRunner.run_epoch(test_runner, train_runner, tracker, epoch_id)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{config.params.epoch_count}]",
                f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        # reset
        train_runner.reset()
        test_runner.reset()

        # flush the tracker after every epoch for live updates
        tracker.flush()



if __name__ == "__main__":
    # args = argparse.ArgumentParser(description='Papyrus')
    # args.add_argument('-cfg_file', '--config_file', default=None, type=str,
    #                help='config file name in ./config')
    # print(args.parse_args())
    run()
