import sys
import torch
import argparse

# Additional Scripts
from train import TrainTestPipe
from inference import SegInference

torch.cuda.empty_cache()


def main_pipeline(parser):
    device = 'cpu:0'
    if torch.cuda.is_available():
        device = 'cuda:0'

    if parser.mode == 'train':
        ttp = TrainTestPipe(mode="train",
                            dataset_path=parser.dataset_path,
                            model_path=parser.model_path,
                            device=device)

        ttp.train()

    elif parser.mode == "evaluate":
        ttp = TrainTestPipe(mode="evaluate",
                            dataset_path=parser.dataset_path,
                            model_path=parser.model_path,
                            device=device)

        ttp.evaluate()

    elif parser.mode == 'inference':
        inf = SegInference(model_path=parser.model_path,
                           device=device)

        _ = inf.infer(parser.data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'inference'])
    parser.add_argument('--model_path', required=True, type=str, default=None)

    parser.add_argument('--dataset_path', required='train' in sys.argv, type=str, default=None)

    parser.add_argument('--data_path', required='infer' in sys.argv, type=str, default=None)
    parser = parser.parse_args()

    if parser.mode in ['train', 'evaluate']:
        assert parser.dataset_path is not None, 'dataset_path must be defined in training mode!'

    elif parser.mode == 'inference':
        assert parser.data_path is not None, 'data_path must be defined in inference mode!'

    assert parser.model_path is not None, 'model_path must be defined'

    main_pipeline(parser)
