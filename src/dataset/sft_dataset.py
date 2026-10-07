import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from params import DataArguments
from constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import (
    chat_template_uses_reasoning_prefill,
    format_assistant_response,
    get_image_info,
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    get_siglip_image_pixels,
    get_siglip_rotated_pixels,
    get_siglip_video_pixels,
    ROTATE_AUG_DEGREES,
    get_video_info,
    llava_to_openai,
    model_supports_optional_reasoning,
    pad_sequence,
    use_default_system_message,
)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        vision_encoder_id=None,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding

        # Optional per-sample question-type labels for curriculum sampling.
        # `self.qtypes[i]` is the category of `self.list_data_dict[i]`; ids absent
        # from the map get "__unknown__" (the sampler folds those into its
        # fallback category). Only loaded for the curriculum-enabled train set.
        self.qtypes = None
        qtype_map_path = getattr(data_args, "qtype_map_path", None)
        if getattr(data_args, "curriculum_enable", False) and qtype_map_path:
            with open(qtype_map_path, "r") as f:
                qtype_map = json.load(f)
            self.qtypes = [
                qtype_map.get(rec.get("id"), "__unknown__") for rec in self.list_data_dict
            ]
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes
        # Opt-in: feed each image as a 4-view rotation stack on the SigLIP path.
        self.rotate_augment = getattr(data_args, "rotate_augment", False)

        self.model_type, self.image_patch_size, self.return_video_metadata = get_qwen_multimodal_settings(
            self.model_id
        )
        self.reasoning_supported = chat_template_uses_reasoning_prefill(self.processor, self.model_type)
        self.use_reasoning_prefill = self.data_args.enable_reasoning and self.reasoning_supported
        self.optional_reasoning_supported = model_supports_optional_reasoning(self.model_type)
        if self.data_args.enable_reasoning and not self.reasoning_supported:
            raise ValueError(
                f"`enable_reasoning` is only supported for Qwen3-VL Thinking or Qwen3.5 models with official reasoning chat templates. "
                f"Current model_type={self.model_type!r} does not qualify."
            )

        # Swapped fixed-resolution vision encoder (SigLIP2 or LemonFM): bypass the Qwen pixel/grid
        # contract and produce fixed square images + a manually-expanded, count-matched set of
        # <|image_pad|> tokens. (`is_siglip` predates LemonFM; it means "swapped encoder path".)
        self.is_siglip = vision_encoder_id is not None
        if self.is_siglip:
            from transformers import AutoConfig

            from model.vision_encoder import load_vision_encoder_assets

            self.siglip_image_processor, sig_vcfg = load_vision_encoder_assets(vision_encoder_id)
            qwen_vcfg = getattr(AutoConfig.from_pretrained(self.model_id), "vision_config", None)
            self.spatial_merge_size = getattr(qwen_vcfg, "spatial_merge_size", 2) if qwen_vcfg else 2
            self.grid_hw = sig_vcfg.image_size // sig_vcfg.patch_size
            merged_hw = self.grid_hw // self.spatial_merge_size
            self.tokens_per_image = merged_hw * merged_hw
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            self.video_token_id = self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_TOKEN)

    def __len__(self):
        return len(self.list_data_dict)

    def _getitem_siglip(self, i) -> Dict[str, torch.Tensor]:
        """SigLIP2 variant of __getitem__.

        Produces fixed [n, 3, S, S] pixel tensors, a synthesized grid_thw of (T, grid_hw, grid_hw)
        per media, and manually expands each <|image_pad|>/<|video_pad|> to
        ``tokens_per_image * num_frames`` copies so the placeholder count matches the vision
        features the SigLIP tower emits (and the M-RoPE layout from the synthesized grid).
        """
        sources = self.list_data_dict[i]
        is_video = "video" in sources
        tokenizer = self.processor.tokenizer
        gh = self.grid_hw
        tpi = self.tokens_per_image

        media_pixels = []   # list of [rows, 3, S, S]
        media_grids = []    # list of [1, 3] rows = (T, gh, gh)
        media_frames = []   # per-media frame count T, in appearance order
        pixel_key = grid_key = None

        if "image" in sources:
            pixel_key, grid_key = "pixel_values", "image_grid_thw"
            files = sources["image"]
            folder = self.data_args.image_folder
            if isinstance(files, str):
                files = [files]
            for f in files:
                if not os.path.exists(f) and not f.startswith("http"):
                    f = os.path.join(folder, f)
                if self.rotate_augment:
                    # I, rot90, rot180, rot270 as a 4-"frame" stack of one image: a single
                    # <|image_pad|> expands to tokens_per_image * 4, matching the 4 rows.
                    n = len(ROTATE_AUG_DEGREES)
                    media_pixels.append(get_siglip_rotated_pixels(f, self.siglip_image_processor))
                    media_grids.append(torch.tensor([[n, gh, gh]]))
                    media_frames.append(n)
                else:
                    media_pixels.append(get_siglip_image_pixels(f, self.siglip_image_processor))
                    media_grids.append(torch.tensor([[1, gh, gh]]))
                    media_frames.append(1)
        elif "video" in sources:
            pixel_key, grid_key = "pixel_values_videos", "video_grid_thw"
            files = sources["video"]
            folder = self.data_args.image_folder
            if isinstance(files, str):
                files = [files]
            for f in files:
                if not os.path.exists(f) and not f.startswith("http"):
                    f = os.path.join(folder, f)
                px, num_frames = get_siglip_video_pixels(
                    f, self.siglip_image_processor, fps=self.fps, nframes=self.nframes
                )
                media_pixels.append(px)
                media_grids.append(torch.tensor([[num_frames, gh, gh]]))
                media_frames.append(num_frames)

        conv = copy.deepcopy(llava_to_openai(sources["conversations"], is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_mm_token_type_ids = []
        media_ptr = 0

        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(self.model_type):
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            sys_ids = tokenizer(system_message, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
            all_input_ids.append(sys_ids)
            all_labels.append(torch.full_like(sys_ids, IGNORE_INDEX))
            all_mm_token_type_ids.append(torch.zeros_like(sys_ids))

        for j in range(0, len(conv), 2):
            user_turn = conv[j]
            gpt_response = conv[j + 1]
            has_reasoning = isinstance(gpt_response.get("reasoning"), str) and gpt_response["reasoning"].strip()
            if self.data_args.enable_reasoning and self.reasoning_supported and not self.optional_reasoning_supported and not has_reasoning:
                raise ValueError(
                    "When `--enable_reasoning True` is used with Qwen3-VL Thinking, every assistant turn must include a non-empty `reasoning` field."
                )
            use_reasoning_prefill = self.use_reasoning_prefill and has_reasoning
            use_closed_think_prefill = self.optional_reasoning_supported and not use_reasoning_prefill
            assistant_prefill, assistant_content = format_assistant_response(
                gpt_response["content"],
                gpt_response.get("reasoning"),
                enable_reasoning=self.data_args.enable_reasoning,
                use_reasoning_prefill=use_reasoning_prefill,
                use_closed_think_prefill=use_closed_think_prefill,
            )

            user_input = (
                f"{DEFAULT_IM_START_TOKEN}{user_turn['role']}\n{user_turn['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{assistant_prefill}"
            )
            gpt_text = f"{assistant_content}{DEFAULT_IM_END_TOKEN}\n"

            prompt_ids = tokenizer(user_input, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)

            # Expand each media placeholder to (tokens_per_image * frames) copies + type ids.
            exp_ids, exp_types = [], []
            for tid in prompt_ids.tolist():
                if tid == self.image_token_id or tid == self.video_token_id:
                    n_tokens = tpi * media_frames[media_ptr]
                    media_ptr += 1
                    exp_ids.extend([tid] * n_tokens)
                    exp_types.extend([1 if tid == self.image_token_id else 2] * n_tokens)
                else:
                    exp_ids.append(tid)
                    exp_types.append(0)
            prompt_ids = torch.tensor(exp_ids, dtype=torch.long)
            prompt_types = torch.tensor(exp_types, dtype=torch.long)

            response_ids = tokenizer(gpt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)

            all_input_ids.append(torch.cat([prompt_ids, response_ids]))
            all_labels.append(torch.cat([torch.full((prompt_ids.size(0),), IGNORE_INDEX, dtype=torch.long), response_ids]))
            all_mm_token_type_ids.append(torch.cat([prompt_types, torch.zeros_like(response_ids)]))

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        mm_token_type_ids = torch.cat(all_mm_token_type_ids, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mm_token_type_ids=mm_token_type_ids,
        )
        if pixel_key and media_pixels:
            data_dict[pixel_key] = torch.cat(media_pixels, dim=0)
            data_dict[grid_key] = torch.cat(media_grids, dim=0)
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if getattr(self, "is_siglip", False):
            return self._getitem_siglip(i)

        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                        image_file, 
                        self.image_min_pixel, 
                        self.image_max_pixel, 
                        self.image_resized_w, 
                        self.image_resized_h, 
                        self.image_patch_size
                    )
                images.append(image_input)

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file, 
                    self.video_min_pixel, 
                    self.video_max_pixel, 
                    self.video_resized_w, 
                    self.video_resized_h, 
                    self.data_args.fps,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata
                )
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_mm_token_type_ids = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0
        
        # Qwen2-VL uses a default system message so I've added this.
        # Qwen3-Vl does not use a system message by default.
        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(self.model_type):
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
            all_mm_token_type_ids.append(torch.zeros_like(system_message_input_ids, dtype=torch.long).squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]
            has_reasoning = isinstance(gpt_response.get("reasoning"), str) and gpt_response["reasoning"].strip()
            if self.data_args.enable_reasoning and self.reasoning_supported and not self.optional_reasoning_supported and not has_reasoning:
                raise ValueError(
                    "When `--enable_reasoning True` is used with Qwen3-VL Thinking, every assistant turn must include a non-empty `reasoning` field."
                )
            use_reasoning_prefill = self.use_reasoning_prefill and has_reasoning
            use_closed_think_prefill = self.optional_reasoning_supported and not use_reasoning_prefill
            assistant_prefill, assistant_content = format_assistant_response(
                gpt_response["content"],
                gpt_response.get("reasoning"),
                enable_reasoning=self.data_args.enable_reasoning,
                use_reasoning_prefill=use_reasoning_prefill,
                use_closed_think_prefill=use_closed_think_prefill,
            )

            user_input = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{assistant_prefill}"
            )
            gpt_response = f"{assistant_content}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                num_images = user_input.count(DEFAULT_IMAGE_TOKEN)
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input:
                num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                # Slice the videos list to get the videos for the current turn.
                videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                if self.model_type == "qwen2_5_vl":
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt', 
                        **video_kwargs
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                elif self.model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
                    videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                    video_datas_for_turn, video_metadatas_for_turn = zip(*videos_for_this_turn)
                    video_datas_for_turn = list(video_datas_for_turn)
                    video_metadatas_for_turn = list(video_metadatas_for_turn)

                    inputs = processor(
                        text=[user_input],
                        images=images,
                        videos=video_datas_for_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors='pt',
                        **video_kwargs,
                        video_metadata=video_metadatas_for_turn,
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                else:
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt'
                    )
                    prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
                prompt_mm_token_type_ids = torch.zeros_like(prompt_input_ids, dtype=torch.long)

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            response_mm_token_type_ids = torch.zeros_like(response_input_ids, dtype=torch.long)

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            mm_token_type_ids = torch.cat([prompt_mm_token_type_ids, response_mm_token_type_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_mm_token_type_ids.append(mm_token_type_ids)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        mm_token_type_ids = torch.cat(all_mm_token_type_ids, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mm_token_type_ids=mm_token_type_ids,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_mm_token_type_ids = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_mm_token_type_ids.append(example["mm_token_type_ids"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        mm_token_type_ids = pad_sequence(batch_mm_token_type_ids, padding_side='right', padding_value=0)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'mm_token_type_ids': mm_token_type_ids,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

def make_supervised_data_module(model_id, processor, data_args, vision_encoder_id=None):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id,
        vision_encoder_id=vision_encoder_id,
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id,
              vision_encoder_id=vision_encoder_id,
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
