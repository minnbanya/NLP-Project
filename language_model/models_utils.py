from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
import torch

def load_model(model_name, temp = 0, rep = 1.5):
    if model_name == 't5':
        return t5_model(temp, rep)
    elif model_name == 'gpt2':
        return gpt2_model(temp, rep)
    elif model_name == 'flan_t5':
        return flan_t5_model(temp, rep)
    elif model_name == 'flan_t5_large':
        return flan_t5_large_model(temp, rep)

def t5_model(temp = 0, rep = 1.5):
    model_id = './models/fastchat-t5-3b-v1.0/'

    tokenizer = AutoTokenizer.from_pretrained(
        model_id)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    bitsandbyte_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config = bitsandbyte_config, #caution Nvidia
        device_map = 'auto',
        load_in_8bit = True
    )

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = 100,
        model_kwargs = {
            "temperature" : temp,
            "repetition_penalty": rep
        }
    )

    llm = HuggingFacePipeline(pipeline = pipe)

    return llm

def gpt2_model(temp = 0, rep = 1.5):
    model_id = 'models/gpt2-span-head-few-shot-k-16-finetuned-squad-seed-0/'

    tokenizer = AutoTokenizer.from_pretrained(
        model_id)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    bitsandbyte_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bitsandbyte_config,
        device_map='cuda:0',
        load_in_8bit=True
    )

    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        max_new_tokens=100,
        model_kwargs={
            "temperature": temp,
            "repetition_penalty": rep
        }
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

def flan_t5_model(temp = 0, rep = 1.5):
    model_id = './models/flan-t5-base/'

    tokenizer = AutoTokenizer.from_pretrained(
        model_id)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    bitsandbyte_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config = bitsandbyte_config, #caution Nvidia
        device_map = 'auto',
        load_in_8bit = True
    )

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = 100,
        model_kwargs = {
            "temperature" : temp,
            "repetition_penalty": rep
        }
    )

    llm = HuggingFacePipeline(pipeline = pipe)

    return llm

def flan_t5_large_model(temp = 0, rep = 1.5):
    model_id = './models/flan-t5-large/'

    tokenizer = AutoTokenizer.from_pretrained(
        model_id)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    bitsandbyte_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config = bitsandbyte_config, #caution Nvidia
        device_map = 'auto',
        load_in_8bit = True
    )

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = 100,
        model_kwargs = {
            "temperature" : temp,
            "repetition_penalty": rep
        }
    )

    llm = HuggingFacePipeline(pipeline = pipe)

    return llm