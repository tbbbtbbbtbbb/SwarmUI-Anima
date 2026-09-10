import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
from comfy.ldm.anima.model import Anima
from comfy.ldm.modules.attention import optimized_attention
from comfy.weight_adapter.lora import LoRAAdapter
import folder_paths


def load_weights(folder, name):
    path = folder_paths.get_full_path_or_raise(folder, name)
    with safe_open(path, framework='pt', device='cpu') as file:
        return {key: file.get_tensor(key) for key in file.keys()}, file.metadata() or {}


def apply_lora(model, weights, prefix, strength, alpha=None):
    patches = {}
    for key, down in weights.items():
        if key.startswith(prefix) and key.endswith('.lora_A.weight'):
            name = key[len(prefix):-len('.lora_A.weight')]
            up = weights[key.replace('.lora_A.weight', '.lora_B.weight')]
            patches[f'diffusion_model.{name}.weight'] = LoRAAdapter(
                [], (up, down, float(alpha) if alpha is not None else down.shape[0], None, None, None))
    if not patches:
        raise ValueError('Anima adapter has no supported LoRA weights.')
    applied = model.add_patches(patches, strength_patch=strength)
    missing = patches.keys() - set(applied)
    if missing:
        raise ValueError(f'Anima adapter does not match this checkpoint: {sorted(missing)[0]}')


def check_anima(model):
    if not isinstance(model.model.diffusion_model, Anima):
        raise ValueError('SwarmAnima requires an Anima checkpoint.')


class ReferenceBlock(nn.Module):
    def __init__(self):
        super().__init__()
        linear = comfy.ops.manual_cast.Linear
        self.ip_k_proj = linear(768, 2048)
        self.ip_v_proj = linear(768, 2048)
        self.adaln_ip = nn.Sequential(nn.SiLU(), linear(2048, 2048))


class ReferenceWeights(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.blocks = nn.ModuleList(ReferenceBlock() for _ in range(count))


class ReferencePatch:
    def __init__(self, adapter, features, strength, sigma_start, sigma_end):
        self.adapter = adapter
        self.features = features
        self.strength = strength
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def models(self):
        return [self.adapter]

    def __call__(self, args):
        options = args['transformer_options']
        sigma = float(options['sigmas'].max())
        options['model_patch_data'][self] = {'active': self.sigma_end <= sigma <= self.sigma_start}
        return args

    def capture(self, q, k, v, extra_options, **kwargs):
        state = extra_options['model_patch_data'][self]
        if state['active']:
            state['query'] = q
        return {}

    def wrap(self, block, index, original):
        def forward(x, emb, context, **kwargs):
            result = original(x, emb, context, **kwargs)
            options = kwargs['transformer_options']
            state = options['model_patch_data'][self]
            if not state['active']:
                return result
            query = state.pop('query')
            labels = options.get('cond_or_uncond', [0])
            if all(label == 1 for label in labels):
                return result
            b, t, h, w, d = result.shape
            heads, head_dim = block.cross_attn.n_heads, block.cross_attn.head_dim
            weights = self.adapter.model.blocks[index]
            features = self.features.to(device=query.device, dtype=query.dtype)
            q = block.cross_attn.q_proj(query).reshape(b, -1, heads, head_dim)
            q = block.cross_attn.q_norm(q).transpose(1, 2)
            k = weights.ip_k_proj(features).reshape(1, -1, heads, head_dim).transpose(1, 2).expand(b, -1, -1, -1)
            v = weights.ip_v_proj(features).reshape(1, -1, heads, head_dim).transpose(1, 2).expand(b, -1, -1, -1)
            out = optimized_attention(q, k, v, heads, skip_reshape=True, transformer_options=options)
            gate = weights.adaln_ip(emb).reshape(b, t, 1, 1, d)
            mask = torch.tensor([label != 1 for label in labels], device=result.device, dtype=result.dtype)
            mask = mask.repeat_interleave(b // len(labels)).reshape(b, 1, 1, 1, 1)
            return result + self.strength * mask * gate * out.reshape(b, t, h, w, d)
        return forward


class SwarmAnimaReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'features': ('ANIMA_FEATURES',),
                             'strength': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 2}),
                             'start': ('FLOAT', {'default': 0.0, 'min': 0, 'max': 1}),
                             'end': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 1})}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'SwarmUI/Anima'

    def apply(self, model, features, strength, start, end):
        check_anima(model)
        if strength == 0:
            return (model,)
        if not 0 <= start < end <= 1:
            raise ValueError('Reference Start must be less than Reference End, both between 0 and 1.')
        if tuple(features.shape) != (1, 1024, 768):
            raise ValueError('Expected one SigLIP2 reference with shape [1, 1024, 768].')
        weights, metadata = load_weights('ipadapter', 'anima-character-reference.safetensors')
        if metadata.get('ip_norm_keys', 'False') != 'False' or metadata.get('ip_inject_before_mlp', 'False') != 'False':
            raise ValueError('Use the supported Anima Character Reference 10 adapter.')
        dit = model.model.diffusion_model
        module = ReferenceWeights(len(dit.blocks))
        module.load_state_dict({k: v for k, v in weights.items() if k.startswith('blocks.')}, strict=True, assign=True)
        adapter = comfy.model_patcher.ModelPatcher(module, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device())
        patched = model.clone()
        apply_lora(patched, weights, 'lora.base_model.model.', strength, metadata.get('lora_alpha'))
        sampling = model.get_model_object('model_sampling')
        patch = ReferencePatch(adapter, features, strength, sampling.percent_to_sigma(start), sampling.percent_to_sigma(end))
        patched.set_model_patch(patch, 'post_input')
        patched.set_model_attn2_patch(patch.capture)
        for i, block in enumerate(dit.blocks):
            name = f'diffusion_model.blocks.{i}.forward'
            patched.add_object_patch(name, patch.wrap(block, i, model.get_model_object(name)))
        return (patched,)


class PosePatch:
    def __init__(self, latent, weight, bias, strength):
        self.latent = latent
        self.weight = weight
        self.bias = bias
        self.strength = strength

    def __call__(self, args):
        output = args['img']
        control = self.latent.to(device=output.device, dtype=output.dtype)
        if control.ndim == 4:
            control = control.unsqueeze(2)
        b, c, t, h, w = control.shape
        if (t, h // 2, w // 2) != tuple(output.shape[1:4]):
            raise ValueError('Pose Image dimensions must match the generation dimensions.')
        tokens = control.reshape(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, t, h // 2, w // 2, c * 4)
        tokens = F.linear(tokens, self.weight.to(output), self.bias.to(output))
        args['img'] = output + self.strength * tokens
        return args


class SwarmAnimaPoseApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'control_latent': ('LATENT',),
                             'strength': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 2})}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'SwarmUI/Anima'

    def apply(self, model, control_latent, strength):
        check_anima(model)
        if strength == 0:
            return (model,)
        weights, _ = load_weights('loras', 'anima-pose-preview2.safetensors')
        patched = model.clone()
        apply_lora(patched, weights, 'diffusion_model.', strength)
        control = control_latent['samples']
        if control.ndim == 4:
            control = control.unsqueeze(2)
        # The control embedder consumes the same normalized latent space as the DiT.
        control = model.model.process_latent_in(control)
        patched.set_model_patch(PosePatch(control,
            weights['diffusion_model.control_embedder.proj.weight'],
            weights['diffusion_model.control_embedder.proj.bias'], strength), 'post_input')
        return (patched,)
