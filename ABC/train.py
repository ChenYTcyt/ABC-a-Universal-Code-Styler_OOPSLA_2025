from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from config import get_default_args
from dataset_utils import get_tokenized_dataset


if __name__ == "__main__":
    args = get_default_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if args.peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
            # target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
    
    train_dataset = get_tokenized_dataset(args, tokenizer, 'train')
    train_len = len(train_dataset)
    eval_dataset = get_tokenized_dataset(args, tokenizer, 'eval')

    lr = float(args.lr)
    eval_steps = int(train_len / args.batch_size * 0.2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        save_total_limit=1,
        load_best_model_at_end=True,
        save_strategy='steps',
        save_steps=eval_steps,
        eval_strategy='steps',
        eval_steps=eval_steps,
        logging_steps=20,
        save_safetensors=False,
        learning_rate=lr,
        warmup_ratio=0.1,
        report_to='tensorboard',
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()