# llama wrapper adapted from https://github.com/nrimsky/CAA/tree/main

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from safetensors.torch import load_file
from collections import defaultdict

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False
    #https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L422
    def forward(self, *args, **kwargs):
        if self.act_as_identity:
            #print(kwargs)
            kwargs['attention_mask'] += kwargs['attention_mask'][0, 0, 0, 1]*torch.tril(torch.ones(kwargs['attention_mask'].shape,
                                                                                                   dtype=kwargs['attention_mask'].dtype,
                                                                                                   device=kwargs['attention_mask'].device),
                                                                                        diagonal=-1)
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None
        self.act_as_identity = False

class MLPWrapper(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.up_proj = mlp.up_proj
        self.gate_proj = mlp.gate_proj
        self.act_fn = mlp.act_fn
        self.down_proj = mlp.down_proj
        self.neuron_interventions = {}
        self.post_activation = None
    
    def forward(self, x):
        post_activation = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        self.post_activation = post_activation.detach().cpu()
        output = self.down_proj(post_activation)
        if len(self.neuron_interventions) > 0:
            print('performing intervention: mlp_neuron_interventions')
            for neuron_idx, mean in self.neuron_interventions.items():
                output[:, :, neuron_idx] = mean
        return output
    
    def reset(self):
        self.neuron_interventions = {}
    
    def freeze_neuron(self, neuron_idx, mean):
        self.neuron_interventions[neuron_idx] = mean

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = self.block.post_attention_layernorm
        self.add_to_last_tensor = None
        self.output = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if self.add_to_last_tensor is not None:
            print('performing intervention: add_to_last_tensor')
            output[0][:, -1, :] += self.add_to_last_tensor
        self.output = output[0]
        return output

    def mlp_freeze_neuron(self, neuron_idx, mean):
        self.block.mlp.freeze_neuron(neuron_idx, mean) 

    def block_add_to_last_tensor(self, tensor):
        self.add_to_last_tensor = tensor

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.block.mlp.reset()
        self.add_to_last_tensor = None

    def get_attn_activations(self):
        return self.block.self_attn.activations

class LlamaHelper:
    def __init__(self, dir='/dlabdata1/llama2_hf/Llama-2-7b-hf', 
                 layer_idcs=None,
                 max_length=512,
                 hf_token=None, 
                 device=None, 
                 load_in_8bit=True, 
                 use_embed_head=False, 
                 device_map='auto'):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(dir, use_auth_token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.model = AutoModelForCausalLM.from_pretrained(dir, use_auth_token=hf_token,
                                                          device_map=device_map,
                                                          load_in_8bit=load_in_8bit)
        self.use_embed_head = True
        W = list(self.model.model.embed_tokens.parameters())[0].detach()
        self.head_embed = torch.nn.Linear(W.size(1), W.size(0), bias=False)
        self.head_embed.to(W.dtype)
        self.norm = self.model.model.norm
        with torch.no_grad():
            self.head_embed.weight.copy_(W) 
        self.head_embed.to(self.model.device)
        self.head_unembed = self.model.lm_head
        #self.model = self.model.to(self.device)
        self.device = next(self.model.parameters()).device
        if use_embed_head:
            head = self.head_embed
        else:
            head = self.head_unembed

        if layer_idcs is None:
            self.layer_idcs = torch.arange(len(self.model.model.layers))
        else:
            self.layer_idcs = layer_idcs

        for i in self.layer_idcs:
            layer = self.model.model.layers[i]
            self.model.model.layers[i] = BlockOutputWrapper(layer, head, self.model.model.norm)
    
    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def sample_next_token(self, logits, temperature=1.0):
        assert temperature >= 0, "temp must be geq 0"
        if temperature == 0:
            return self._sample_greedy(logits)
        return self._sample_basic(logits/temperature)
        
    def _sample_greedy(self, logits):
        return logits.argmax().item()

    def _sample_basic(self, logits):
        return torch.distributions.categorical.Categorical(logits=logits).sample().item()
    
    def get_logits(self, prompts):
        #inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, padding_side='left')
        # Initialize a list to hold the padded input_ids and attention_mask tensors
        padded_input_ids_list = []
        padded_attention_mask_list = []
        
        # Tokenize each prompt individually
        max_length = self.max_length  # Define the desired max length
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_length)
            input_ids = inputs['input_ids'][0]  # Extract the tensor; assume batch_size=1
            attention_mask = inputs['attention_mask'][0]  # Extract the tensor; assume batch_size=1
            
            # Calculate how much padding is needed to achieve the max length
            padding_length = max_length - len(input_ids)
            
            # Apply left-side padding manually
            padded_input_ids = torch.cat([torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long), input_ids])
            padded_attention_mask = torch.cat([torch.zeros((padding_length,), dtype=torch.long), attention_mask])
            
            #print(attention_mask)
            #print(padded_attention_mask)
            # Append the padded tensors to the lists
            padded_input_ids_list.append(padded_input_ids)
            padded_attention_mask_list.append(padded_attention_mask)
        
        # Stack the list of tensors to create batched inputs
        padded_input_ids_batch = torch.stack(padded_input_ids_list)
        padded_attention_mask_batch = torch.stack(padded_attention_mask_list)
        
        # Prepare the batched inputs for the model
        padded_inputs = {'input_ids': padded_input_ids_batch, 'attention_mask': padded_attention_mask_batch}
    
        with torch.no_grad():
          logits = self.model(**padded_inputs).logits
          return logits

    def set_add_to_last_tensor(self, layer, tensor):
      print('setting up intervention: add tensor to last soft token')
      self.model.model.layers[layer].block_add_to_last_tensor(tensor)

    def reset_all(self):
        for i in self.layer_idcs:
            layer = self.model.model.layers[i]
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(indices.detach().cpu().numpy().tolist(), tokens, probs_percent)))


    def latents_all_layers(self, text, return_attn_mech=False, return_intermediate_res=False, return_mlp=False, return_mlp_post_activation=False, return_block=True):
        if return_attn_mech or return_intermediate_res or return_mlp or return_mlp_post_activation:
            raise NotImplemented("not implemented")
        self.get_logits(text)
        tensors = []
        if return_block:
            for i in self.layer_idcs:
                layer = self.model.model.layers[i]
                latents = layer.output.detach().cpu()
                latents = latents.unsqueeze(0)
                tensors += [latents]
        return torch.cat(tensors, dim=0)