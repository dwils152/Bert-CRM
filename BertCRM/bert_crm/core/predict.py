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
        new_state_dict = {k.replace('module.', '')
                                    : v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    return tokenizer, model


def predict_crf(model, data_loader, device, rank):
    model.eval()
    file_name = f'predictions_with_coordinates_rank_{rank}.txt'

    with torch.no_grad(), open(file_name, 'w+') as f:
        for batch in tqdm_wrapper(data_loader, rank, desc="Predicting"):
            input_ids, attention_mask, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            predictions, _ = model(input_ids, attention_mask)
            # print(len(predictions[0]))

            predictions = torch.tensor(
                [item for sublist in predictions for item in sublist], device=device)

            # Flatten coordinates to match predictions' shape
            coordinates = [item for sublist in coordinates for item in sublist]

            # print(f'length of predictions: {len(predictions)}')
            # print(f'length of coordinates: {len(coordinates)}')

            # Flatten input_ids and attention_mask for processing
            flat_input_ids = input_ids.view(-1)
            flat_attention_mask = attention_mask.view(-1)

            # Mask for active positions (not padded)
            active_positions = flat_attention_mask == 1

            # Create a mask to exclude specific tokens. Adjust token values as necessary.
            exclude_tokens = (flat_input_ids != 1) & (
                flat_input_ids != 2) & (flat_input_ids != 3)

            # Combine masks for final filtering
            final_mask = active_positions & exclude_tokens

            # Apply final mask to predictions and coordinates
            active_predictions = predictions[final_mask]
            active_coordinates = [coordinates[i]
                                  for i in range(len(coordinates)) if final_mask[i]]

            # print(f'length of active predictions: {len(active_predictions)}')
            # print(f'length of active coordinates: {len(active_coordinates)}')
            # sys.exit()

            # Write results directly to file
            for coord, pred in zip(active_coordinates, active_predictions.tolist()):
                f.write(f"{coord}: {pred}\n")


def predict_no_crf(model, data_loader, device, rank):
    model.eval()
    file_name = f'predictions_with_coordinates_rank_{rank}.txt'

    with torch.no_grad(), open(file_name, 'w+') as f:
        for batch in tqdm_wrapper(data_loader, rank, desc="Predicting"):
            input_ids, attention_mask, coordinates = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            logits = model.forward(input_ids, attention_mask)
            logits = logits.view(-1, 2)

            # active_loss = attention_mask.view(-1) == 1
            # active_logits = logits[active_loss]
            coordinates = [item for sublist in coordinates for item in sublist]
            # print(coordinates)

            # Flatten input_ids and attention_mask for processing
            flat_input_ids = input_ids.view(-1)
            flat_attention_mask = attention_mask.view(-1)

            # Mask for active positions (not padded)
            active_positions = flat_attention_mask == 1

            # Create a mask to exclude specific tokens. Adjust token values as necessary.
            exclude_tokens = (flat_input_ids != 1) & (
                flat_input_ids != 2) & (flat_input_ids != 3) & (flat_input_ids != 0)

            # Combine masks for final filtering
            final_mask = active_positions & exclude_tokens

            active_logits = logits[final_mask]
            preds = torch.argmax(active_logits, dim=1)
            probabilities = torch.softmax(active_logits, dim=1)
            pos_proba = probabilities[:, 1]

            # Apply final mask to predictions and coordinates
            active_coordinates = [coordinates[i]
                                  for i in range(len(coordinates)) if final_mask[i]]

            # print(f'length of active predictions: {len(preds)}')
            # print(f'length of active probabilities: {len(probabilities)}')
            # print(f'length of active coordinates: {len(active_coordinates)}')
            # sys.exit()

            # Write results directly to file
            for (coord, pred, prob) in zip(active_coordinates, preds.tolist(), pos_proba.tolist()):
                f.write(f"{coord}: {pred}\t{prob}\n")


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
