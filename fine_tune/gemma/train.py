from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os


max_seq_length = 2048 
dtype = None 
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-2b-it",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 256,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None,
)

gemma_prompt = """<start_of_turn>user
{}
<end_of_turn>
<start_of_turn>model
{}
<end_of_turn>
"""


EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = gemma_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

dataset = load_dataset("trishonc/agent_buff", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args = TrainingArguments(
        output_dir = "main",
        num_train_epochs = 1,
        per_device_train_batch_size = 16,
        weight_decay = 1e-3,
        warmup_steps = 0.01,
        logging_steps = 10,
        logging_dir="logs",
        save_strategy = "steps",
        evaluation_strategy= "steps",
        eval_steps = 1505,
        save_steps = 1505,
        learning_rate = 1e-4,
        bf16 = is_bfloat16_supported,
        lr_scheduler_type = 'cosine',
        seed = 3407, 
        ),
)

trainer_stats = trainer.train()

# FastLanguageModel.for_inference(model)
# inputs = tokenizer(
# [
#     gemma_prompt.format(
#         "Continue the fibonnaci sequence: 1, 1, 2, 3, 5,", 
#         "",
#     )
# ], return_tensors = "pt").to("cuda")

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


model.save_pretrained("lora_model") 
tokenizer.save_pretrained("lora_model")

# model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged("trishonc/gemma-2-2b-it-buffed", tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))