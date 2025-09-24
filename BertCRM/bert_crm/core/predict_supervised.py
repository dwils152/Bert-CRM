import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import bert_crm
from transformers import AutoTokenizer
from bert_crm.models.BertForTokenClassification import BertForTokenClassification as BertTok
from bert_crm.models.BertCRFForTokenClassification import BertCRFForTokenClassification as BertTokCRF
from bert_crm.models.UnsupervisedDataset import UnsupervisedDataset
from bert_crm.models.SupervisedDataset import SupervisedDataset
import argparse
from bert_crm.utils.utils import (
    setup,
    tqdm_wrapper,
    unsupervised_collate_fn,
    supervised_collate_fn, 
    cleanup
)
import sys
from tqdm import tqdm


def setup_model(model_path, use_ddp=False):
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertTok(args.base_model, num_labels=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if use_ddp:
        new_state_dict = {
            'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}
    else:
        new_state_dict = {k.replace('module.', '')
                                    : v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    return tokenizer, model


def main(args):

        model = setup_model(args.model_path, use_ddp=False)[1]
        rank = int(os.environ['LOCAL_RANK'])
        device = 'cuda'
        model = model.to(device)
        dataset = SupervisedDataset(args.data_path, 1024)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=supervised_collate_fn)

        model.test_model(dataloader, device, rank)
        #cleanup()

        


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, required=True)
    argparser.add_argument("--data_path", type=str, required=True)
    argparser.add_argument("--base_model", type=str,
                           default='zhihan1996/DNABERT-2-117M')

    args = argparser.parse_args()
    main(args)
