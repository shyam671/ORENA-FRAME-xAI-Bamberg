import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from model.load_model import get_qwen_vl_generation_backbone, load_qwen_vl_generation_model
from trainer import QwenSFTTrainer
from dataset import make_supervised_data_module
from params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    backbone = get_qwen_vl_generation_backbone(model)
    vision_tower = backbone.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = backbone.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # Per-encoder override for a multi: fusion tower: keep only the named sub-encoders trainable.
    # Runs after the whole-tower pass above and before the merger, which it never touches.
    if training_args.train_vision_encoders:
        if not hasattr(vision_tower, "set_trainable_encoders"):
            raise ValueError("`train_vision_encoders` requires a multi: fusion tower.")
        keep = {n.strip() for n in training_args.train_vision_encoders.split(",") if n.strip()}
        vision_tower.set_trainable_encoders(keep)
        rank0_print(f"[vision] trainable sub-encoders: {sorted(keep)} of {vision_tower.encoder_names}")

    # Handle merger specifically
    merger_params = backbone.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(backbone.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = backbone.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    backbone = get_qwen_vl_generation_backbone(model)
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = backbone.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    backbone = get_qwen_vl_generation_backbone(model)

    if k_llm and hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        for layer in backbone.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(backbone, "visual") and hasattr(backbone.visual, "blocks"):
        for blk in backbone.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # HF seeds inside Trainer.__init__, which runs long after the model is built, so a freshly
    # initialised merger (the only randomly initialised module in a swapped tower) was drawn from
    # torch's nondeterministically seeded default generator on every run. Seed here instead.
    set_seed(training_args.seed)

    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    if training_args.train_vision_encoders and training_args.freeze_vision_tower:
        raise ValueError("If `train_vision_encoders` is set, `freeze_vision_tower` must be False.")

    if training_args.train_vision_encoders and training_args.unfreeze_topk_vision:
        raise ValueError("`train_vision_encoders` and `unfreeze_topk_vision` conflict: "
                         "`unfreeze_topk_layers` runs afterwards and spans every sub-encoder.")

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = load_qwen_vl_generation_model(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        vision_encoder_id=model_args.vision_encoder_id,
        **bnb_model_from_pretrained_args,
    )

    model.config.use_cache = False
    # Persist so inference re-applies the same input transform (like vision_encoder_id).
    model.config.rotate_augment = data_args.rotate_augment
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

        # A swapped tower (SigLIP2 / LemonFM) is a plain wrapper, so the model-wide GC toggle
        # does not reach it. Enable GC directly on the swapped encoder when it is training.
        backbone = get_qwen_vl_generation_backbone(model)
        visual = getattr(backbone, "visual", None)
        if not training_args.freeze_vision_tower and hasattr(visual, "enable_vision_gradient_checkpointing"):
            visual.enable_vision_gradient_checkpointing()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args,
                                              vision_encoder_id=model_args.vision_encoder_id)

    curriculum_sampler = None
    if data_args.curriculum_enable:
        import json as _json
        from dataset.curriculum_sampler import CurriculumSampler

        train_dataset = data_module["train_dataset"]
        if getattr(train_dataset, "qtypes", None) is None:
            raise ValueError(
                "curriculum_enable=True but the train dataset has no qtypes. "
                "Set --qtype_map_path to a sidecar built by scripts/build_qtype_labels.py."
            )
        with open(data_args.curriculum_schedule) as _f:
            schedule = _json.load(_f)
        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * max(1, training_args.world_size)
        )
        curriculum_sampler = CurriculumSampler(
            qtypes=train_dataset.qtypes,
            schedule=schedule,
            global_batch_size=global_batch_size,
            num_epochs=int(training_args.num_train_epochs),
            seed=data_args.curriculum_seed,
        )
        rank0_print(
            f"[curriculum] global_batch={global_batch_size} "
            f"epochs={int(training_args.num_train_epochs)} "
            f"schedule={data_args.curriculum_schedule}"
        )
        rank0_print("[curriculum] realized phase composition:")
        for phase, comp in curriculum_sampler.phase_composition().items():
            pretty = ", ".join(f"{c}={f:.0%}" for c, f in comp.items())
            rank0_print(f"    {phase}: {pretty}")

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        curriculum_sampler=curriculum_sampler,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
