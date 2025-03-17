import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)


def llm_generate_qwen_coder(model_name, question, parameters):
    model_kwargs = {
        'device_map': 'auto'
    }
    if parameters['load_in_4bit']:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs['quantization_config'] = quantization_config
    else:
        model_kwargs['torch_dtype'] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": " You are an expert in writing performant OpenMP code. You need to write code following the given function signature. Do not include any explanations."},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    default_params = CONFIG['code_generation']['default_parameters']['qwen-coder']

    generated_ids = model.generate(
        **model_inputs,
        **default_params,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response


def llm_generate_smollm2(model_name, question, parameters):
    model_kwargs = {
        'device_map': 'auto'
    }
    if parameters['load_in_4bit']:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs['quantization_config'] = quantization_config
    else:
        model_kwargs['torch_dtype'] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system",
         "content": " You are an expert in writing performant OpenMP code. You need to write code following the given function signature. Do not include any explanations."},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    default_params = CONFIG['code_generation']['default_parameters']['smollm2']

    generated_ids = model.generate(
        **model_inputs,
        **default_params,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response


def llm_generate_codellama(model_name, question, parameters):
    model_kwargs = {
        'device_map': 'auto'
    }
    if parameters['load_in_4bit']:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs['quantization_config'] = quantization_config
    else:
        model_kwargs['torch_dtype'] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system",
         "content": " You are an expert in writing performant OpenMP code. You need to write code following the given function signature. Do not include any explanations."},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    default_params = CONFIG['code_generation']['default_parameters']['codellama']

    generated_ids = model.generate(
        **model_inputs,
        **default_params,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response


def llm_generate_llama(model_name, question, parameters):
    model_kwargs = {
        'device_map': 'auto'
    }
    if parameters['load_in_4bit']:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs['quantization_config'] = quantization_config
    else:
        model_kwargs['torch_dtype'] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system",
         "content": " You are an expert in writing performant OpenMP code. You need to write code following the given function signature. Do not include any explanations."},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    default_params = CONFIG['code_generation']['default_parameters']['llama']

    generated_ids = model.generate(
        **model_inputs,
        **default_params,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response


def llm_generate_deepseek_coder(model_name, question, parameters):
    model_kwargs = {
        'device_map': 'auto'
    }
    if parameters['load_in_4bit']:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs['quantization_config'] = quantization_config
    else:
        model_kwargs['torch_dtype'] = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system",
         "content": " You are an expert in writing performant OpenMP code. You need to write code following the given function signature. Do not include any explanations."},
        {"role": "user", "content": question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    default_params = CONFIG['code_generation']['default_parameters']['deepseek-coder']

    generated_ids = model.generate(
        **model_inputs,
        **default_params,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response


def code_generation(model, question, parameters):
    if model in CONFIG['code_generation']['models'] and model.startswith('deepseek-ai/deepseek-coder'):
        return llm_generate_deepseek_coder(model, question, parameters)
    if model in CONFIG['code_generation']['models'] and model.startswith('Qwen/Qwen2.5-Coder'):
        return llm_generate_qwen_coder(model, question, parameters)
    if model in CONFIG['code_generation']['models'] and model.startswith('HuggingFaceTB/SmolLM2'):
        return llm_generate_smollm2(model, question, parameters)
    if model in CONFIG['code_generation']['models'] and model.startswith('meta-llama/CodeLlama'):
        return llm_generate_codellama(model, question, parameters)
    if model in CONFIG['code_generation']['models'] and model.startswith('meta-llama/Llama'):
        return llm_generate_llama(model, question, parameters)