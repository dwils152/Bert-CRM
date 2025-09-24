from captum.attr import IntegratedGradients
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
import argparse
from bert_crm.utils.utils import (
    setup,
    tqdm_wrapper,
    unsupervised_collate_fn,
    cleanup
)
import sys
from tqdm import tqdm


def setup_model(model_path, use_ddp=False):
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertTokCRF(args.base_model, num_labels=2) if args.use_crf else BertTok(
        args.base_model, num_labels=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if use_ddp:
        new_state_dict = {
            'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}
    else:
        new_state_dict = {k.replace('module.', '')                          : v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    return tokenizer, model


def predict_no_crf(model, data_loader, device, rank):
    model.eval()
    file_name = f'predictions_with_coordinates_rank_{rank}.txt'
    ig = IntegratedGradients(model)  # Initialize Integrated Gradients

    with torch.no_grad(), open(file_name, 'w+') as f:
        for batch in tqdm_wrapper(data_loader, rank, desc="Predicting"):
            input_ids, attention_mask, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            # Prepare inputs for attribution
            inputs = (input_ids, attention_mask)

            # Compute attributions using Integrated Gradients
            attributions = ig.attribute(inputs=input_ids, additional_forward_args=(attention_mask,),
                                        target=1,  # Assuming you are interested in attributions for the positive class
                                        return_convergence_delta=False)

            logits = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            coordinates = [item for sublist in coordinates for item in sublist]
            active_logits = logits.view(-1, model.num_labels)
            # Assuming the second column is the positive class
            pos_proba = torch.softmax(active_logits, dim=1)[:, 1]

            # Extract coordinates and predictions
            for coord, pred, prob, attr in zip(coordinates, preds.view(-1).tolist(), pos_proba.tolist(), attributions.view(-1).tolist()):
                f.write(f"{coord}: {pred}\t{prob}\t{attr}\n")


def main(args):

    if args.is_distributed:

        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        setup(rank, world_size)

        device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        tokenizer, model = setup_model(args.model_path, use_ddp=False)
        model = model.to(f'cuda:{rank}')
        model = DDP(model, device_ids=[rank])
        data_path = args.data_path
        dataset = UnsupervisedDataset(data_path, 1024)
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=32, sampler=sampler, collate_fn=unsupervised_collate_fn)

        if args.use_crf:
            predict_crf(model, dataloader, device, rank)
        else:
            predict_no_crf(model, dataloader, device, rank)

        cleanup()

    else:
        model = setup_model(args.model_path, use_ddp=False)[1]
        device = 'cuda'
        model = model.to(device)
        dataset = UnsupervisedDataset(args.data_path, 1024)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=32, collate_fn=unsupervised_collate_fn)

        # print(dataset[0])
        predict_no_crf(model, dataloader, device, rank='test')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, required=True)
    argparser.add_argument("--data_path", type=str, required=True)
    argparser.add_argument('--use_crf', action='store_true')
    argparser.add_argument('--is_distributed', action='store_true')
    argparser.add_argument("--base_model", type=str,
                           default='/users/dwils152/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/25abaf0bd247444fcfa837109f12088114898d98')
    argparser.add_argument("--local_rank", type=int,
                           help="Local rank. Necessary for using the torch.distributed.launch utility.")

    args = argparser.parse_args()
    main(args)
