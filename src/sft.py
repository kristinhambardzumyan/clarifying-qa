import os
import json
import random
import argparse
from collections import Counter

import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from accelerate import Accelerator, PartialState

from utils import gen_direct_qa_output_prompt, gen_clarify_q_prompt, gen_clarify_a_prompt, gen_qa_output_prompt
from utils import QA_OUTPUT, CLARIFY_Q, CLARIFY_A

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def preprocess(ex, mode):
    sft_examples = []
    if mode == 'gen_clarify_q':
        sft_examples.append(gen_clarify_q_prompt(
            qa_input=ex['question'],
            clarify_q=ex['clarification']['question'],
        ))
    elif mode == 'gen_clarify_a':
        for answer in ex['clarification']['answers']:
            sft_examples.append(gen_clarify_a_prompt(
                qa_input=ex['question'],
                clarify_q=ex['clarification']['question'],
                qa_output=answer['response'],
                clarify_a=answer['answer'],
            ))
    elif mode == 'gen_qa_output':
        for answer in ex['clarification']['answers']:
            sft_examples.append(gen_qa_output_prompt(
                qa_input=ex['question'],
                clarify_q=ex['clarification']['question'],
                clarify_a=answer['answer'],
                qa_output=answer['response'],
            ))
    elif mode == 'gen_direct_qa_output':
        for answer in ex['clarification']['answers']:
            sft_examples.append(gen_direct_qa_output_prompt(
                qa_input=ex['question'],
                qa_output=answer['response'],
            ))
    else:
        raise ValueError

    return sft_examples

def main(args):
    accelerator = Accelerator()

    args.base_model = args.model

    experiment_dir = os.path.join(
        args.output_dir,
        args.model.replace("/", "_"),
        args.model,
        args.mode,
        args.experiment_name,
    )
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=2))

    train_data = {}
    for path in args.train_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            train_data[os.path.split(path)[-1]] = [
                sft_ex for ex in data for sft_ex in preprocess(ex, mode=args.mode)
            ]
    dev_data = {}
    for path in args.dev_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            dev_data[os.path.split(path)[-1]] = [
                sft_ex for ex in data for sft_ex in preprocess(ex, mode=args.mode)
            ]

    if args.test:
        train_data = {k: sorted(v, key=len, reverse=True)[:args.test] for k, v in train_data.items()}
        dev_data = {k: sorted(v, key=len, reverse=True)[:args.test] for k, v in dev_data.items()}


    for name, data in train_data.items():
        with open(os.path.join(experiment_dir, 'train.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')
    for name, data in dev_data.items():
        with open(os.path.join(experiment_dir, 'dev.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')

    train_data = Dataset.from_list([{'text': ex} for data in train_data.values() for ex in data])
    dev_data = {
        name: Dataset.from_list([{'text': ex} for ex in data]) for name, data in dev_data.items()
    }

    per_device_batch_size = args.batch_size
    print(f'Device Count={torch.cuda.device_count()}')
    print(f'Per Device Batch Size={per_device_batch_size}')
    print(f'Grad Accum Steps={args.grad_accum_steps}')
    
    training_args = TrainingArguments(
        output_dir=experiment_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        logging_steps=10,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        report_to='none',
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type='cosine',
    )

    bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.float16,
    )   

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    if args.checkpoint:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(args.checkpoint),
            is_trainable=True
        )
        peft_config = None
    else:
        peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.mode == 'gen_clarify_q':
        response_template = CLARIFY_Q
    elif args.mode == 'gen_clarify_a':
        response_template = CLARIFY_A
    elif args.mode == 'gen_qa_output':
        response_template = QA_OUTPUT
    elif args.mode == 'gen_direct_qa_output':
        response_template = QA_OUTPUT
    else:
        raise ValueError

    response_template_ids = tokenizer.encode(
        response_template,
        add_special_tokens=False
    )
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=data_collator,
        peft_config=peft_config,
    )
    
    trainer.train()

    if accelerator.is_local_main_process:
        save_dir = os.path.join(experiment_dir, 'best_checkpoint')

        with open(os.path.join(experiment_dir, 'log_history.json'), 'w') as f:
            f.write(json.dumps(trainer.state.log_history, indent=2))

        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--experiment_name')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--train_paths', nargs='+')
    parser.add_argument('--dev_paths', nargs='+')
    parser.add_argument('--test', type=int, default=None)
    parser.add_argument('--mode')
    parser.add_argument(
        '--output_dir',
        default='/var/local/mjqzhang/cautious_rlhf/cautious_rlhf/'
    )
    parser.add_argument('--random_seed', type=int, default=88888888)


    # General Training Hyperparameters
    parser.add_argument('--epochs', type=float, default=3.0)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Lora Hyperparameters
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', default="none")


    cli_args = parser.parse_args()
    random.seed(cli_args.random_seed)
    main(cli_args)
