"""Kandinsky 2.2 Gradio UI -- text-to-image plus image tools.

Standalone app, same shape as Disco UI: no Jupyter, no Colab. Run it,
a page opens in your browser, type a prompt, hit Generate, a picture
appears underneath.

How to run (on a Windows PC with an NVIDIA GPU):

    install.bat      (once -- builds a venv, installs torch + packages)
    run.bat          (starts the app)

The app file lives in the kandinsky-ui/ subfolder; run.bat runs it from
the folder above, so pictures land in images_out/ next to run.bat --
one subfolder per batch name, reused across runs, frames numbered
continuously, and settings.txt appended once per run, exactly like
Disco UI.

First Generate takes a while: the prior + decoder models download from
Hugging Face (about 8-12 GB total) and load once, then stay loaded.
Repeat generations reuse what's already loaded. Image-tools pipelines
(img2img reuses the decoder weights, so it costs no extra VRAM;
inpaint and ControlNet load their own weights on first use) download
on first use the same way.

How it works (Diffusers 0.39.0 -- its Kandinsky 2.2 pipeline files
differ from 0.41.0's only in return-type annotations, so the call
arguments here are unaffected):
  1. KandinskyV22PriorPipeline turns the prompt into CLIP image
     embeddings (prior steps, prior guidance scale).
  2. KandinskyV22Pipeline turns those embeddings into the picture
     (decoder steps, guidance scale, width, height, seed).
Guidance here is Kandinsky's scale (default 4.0 for both stages) --
nothing like Disco's 10000-scale clip guidance. Do not carry Disco
numbers over.

Extras, kept in the same plain style as the rest of the app:
size presets, separate prior guidance, several pictures per click,
seed shown in the status line, a Stop button, a Random prompt button
(word banks in code -- no list file to maintain), a session-only
prompt history dropdown, Star batch (copies a batch into
images_out/favorites/), a prompt queue (one prompt per line),
single-level dynamic prompts {a|b|c}, prompt blending (two prompts
and a mix slider), and an Image tools tab (image to image,
inpainting with a painted mask, ControlNet depth from a
user-supplied depth map). No animation, no upscaling.
"""

import os
import random
import re
import shutil
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)  # folder holding run.bat
IMAGES_DIR = os.path.join(ROOT_DIR, "images_out")
FAVORITES_DIR = os.path.join(IMAGES_DIR, "favorites")

PRIOR_ID = "kandinsky-community/kandinsky-2-2-prior"
DECODER_ID = "kandinsky-community/kandinsky-2-2-decoder"
INPAINT_ID = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
CONTROLNET_ID = "kandinsky-community/kandinsky-2-2-controlnet-depth"

DEFAULT_PROMPT = (
    "A lighthouse on a rocky cliff during a storm, waves crashing, "
    "dramatic sky, detailed"
)
DEFAULT_NEGATIVE = "low quality, bad quality"

# Random-prompt word banks, in code on purpose: no external list file
# to maintain, and nothing to keep in sync with the app. Plain
# text-to-image subjects only, so any combination reads as a prompt.
_RANDOM_SUBJECTS = [
    "lighthouse", "old castle", "sailing ship", "mountain cabin",
    "red fox", "steam train", "hot air balloon", "medieval knight",
    "space station", "treehouse", "windmill", "dragon",
    "robot explorer", "harbor town", "desert oasis", "snowy owl",
    "vintage car", "floating island",
]
_RANDOM_PLACES = [
    "on a rocky cliff", "in a misty forest", "beside a quiet lake",
    "above the clouds", "in a desert canyon", "on a snowy mountain pass",
    "in a neon city street", "under the sea", "in a medieval village",
    "on a windswept beach", "inside a grand library",
    "at the edge of a volcano", "in a flower meadow",
    "on a frozen river", "in a hidden valley", "above a waterfall",
]
_RANDOM_STYLES = [
    "oil painting", "watercolor", "detailed digital art",
    "cinematic photo", "storybook illustration", "cyberpunk art",
    "art deco poster", "impressionist painting",
    "fantasy concept art", "steampunk illustration", "anime style",
    "vintage photograph", "pixel art", "surrealist painting",
]
_RANDOM_LIGHTING = [
    "dramatic storm light", "golden hour glow", "soft moonlight",
    "misty morning light", "neon glow at night", "warm sunset colors",
    "cold blue twilight", "dappled sunlight", "eerie fog",
    "bright midday sun", "candlelit warmth", "aurora in the sky",
]

_pipes = None  # (prior, decoder), loaded once on first Generate
_img2img_pipe = None  # built from the decoder's components, no new weights
_inpaint_pipe = None  # loaded once on first Inpaint use
_controlnet_pipe = None  # loaded once on first ControlNet use


def _load_pipes():
    """Load the prior + decoder in fp16 onto CUDA. Raises a plain
    RuntimeError with a readable message if there is no CUDA GPU."""
    global _pipes
    if _pipes is not None:
        return _pipes
    import torch
    from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. Kandinsky UI needs an NVIDIA GPU with "
            "a recent driver -- CPU-only mode is not supported."
        )
    prior = KandinskyV22PriorPipeline.from_pretrained(
        PRIOR_ID, torch_dtype=torch.float16
    )
    prior.to("cuda")
    decoder = KandinskyV22Pipeline.from_pretrained(
        DECODER_ID, torch_dtype=torch.float16
    )
    decoder.to("cuda")
    _pipes = (prior, decoder)
    return _pipes


def _load_img2img(decoder):
    """Build the img2img pipeline from the decoder's own components
    (same unet, scheduler, movq -- Diffusers 0.39.0's
    KandinskyV22Img2ImgPipeline takes exactly those three). No new
    weights are loaded, so this costs no extra VRAM."""
    global _img2img_pipe
    if _img2img_pipe is not None:
        return _img2img_pipe
    from diffusers import KandinskyV22Img2ImgPipeline

    _img2img_pipe = KandinskyV22Img2ImgPipeline(
        unet=decoder.unet, scheduler=decoder.scheduler, movq=decoder.movq
    )
    return _img2img_pipe


def _load_inpaint():
    """Load the dedicated inpaint pipeline in fp16 onto CUDA, once."""
    global _inpaint_pipe
    if _inpaint_pipe is not None:
        return _inpaint_pipe
    import torch
    from diffusers import KandinskyV22InpaintPipeline

    _inpaint_pipe = KandinskyV22InpaintPipeline.from_pretrained(
        INPAINT_ID, torch_dtype=torch.float16
    )
    _inpaint_pipe.to("cuda")
    return _inpaint_pipe


def _load_controlnet():
    """Load the depth ControlNet pipeline in fp16 onto CUDA, once.
    The controlnet-depth repo's UNet takes the depth hint itself, so
    there is no separate controlnet component to wire up."""
    global _controlnet_pipe
    if _controlnet_pipe is not None:
        return _controlnet_pipe
    import torch
    from diffusers import KandinskyV22ControlnetPipeline

    _controlnet_pipe = KandinskyV22ControlnetPipeline.from_pretrained(
        CONTROLNET_ID, torch_dtype=torch.float16
    )
    _controlnet_pipe.to("cuda")
    return _controlnet_pipe


def _safe_batch_name(name):
    """Strip characters Windows forbids in folder names. Falls back to
    a timestamped name when the box is empty."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        name = datetime.now().strftime("kandinsky-%Y%m%d-%H%M%S")
    return name


def _unique_folder(base, root=IMAGES_DIR):
    """<root>/<base>, or <base>_2, <base>_3... if it already exists.
    Used only by Star batch now; generations reuse one folder per
    batch name (see _batch_folder) instead of numbered folders."""
    folder = os.path.join(root, base)
    if not os.path.exists(folder):
        return folder
    i = 2
    while True:
        candidate = "{}_{}".format(folder, i)
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _batch_folder(base):
    """The one folder for a batch name: images_out/<base>, created
    if missing, reused if it already exists. Repeat runs never get
    a _2/_3 folder -- frames just keep numbering upward."""
    return os.path.join(IMAGES_DIR, base)


_FRAME_RE = re.compile(r"^(\d{6})\.png$")


def _next_frame_number(folder):
    """First unused frame number in a folder: one past the highest
    existing 000001-style frame, or 1 for a fresh folder. Only bare
    frame names count, so input_000006.png never affects it."""
    highest = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = _FRAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _frame_range_text(start, count):
    """'frame 000006' for one image, 'frames 000006-000008' for
    several -- used in status lines and settings headers."""
    if count == 1:
        return "frame {:06d}".format(start)
    return "frames {:06d}-{:06d}".format(start, start + count - 1)


def _seeds_for(seed, count):
    """Resolve the seed box to one seed per image. -1 (or empty) draws
    a fresh random seed per image; a fixed seed S gives S, S+1, S+2...
    so a multi-image run with a fixed seed stays reproducible."""
    if seed is None:
        seed = -1
    seed = int(seed)
    if seed < 0:
        return [random.randint(0, 2**31 - 1) for _ in range(count)]
    return [seed + i for i in range(count)]


# Dynamic prompts: single level only, one random pick per group.
# "{a|b|c}" becomes a, b, or c. Groups do not nest; a group with no
# pipe is just its own text with the braces removed.
_TEMPLATE_RE = re.compile(r"\{([^{}]*)\}")


def _expand_templates(prompt):
    """Expand {a|b|c} groups, one random pick each. Plain prompts
    (no braces) come back unchanged."""
    return _TEMPLATE_RE.sub(
        lambda match: random.choice(match.group(1).split("|")), prompt
    )


def _append_settings(folder, header, entries):
    """Append one run's block to settings.txt: a header line (date,
    time, frame range) followed by the run's 'key: value' lines and
    a blank line. Creates the file on first use; earlier runs' blocks
    are never touched. Old single-run files end with a newline, but
    a missing one is repaired so blocks never run together."""
    path = os.path.join(folder, "settings.txt")
    prefix = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + header + "\n")
        handle.write("\n".join(entries) + "\n\n")


def _run_header(start, count):
    """The settings.txt header for one run: date, time, and the
    frame-number range it produced."""
    return "Run {}, {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _frame_range_text(start, count))


def _seed_entry(seeds):
    """The settings.txt seed line: 'seed: X' for one image,
    'seeds: X, Y' for several."""
    if len(seeds) == 1:
        return "seed: {}".format(seeds[0])
    return "seeds: {}".format(", ".join(str(s) for s in seeds))


def _seed_text(seeds):
    """The status-line seed text, same wording as settings.txt."""
    return _seed_entry(seeds)


def _expanded_entries(expanded_list):
    """The settings.txt expanded-prompt line(s): one line for one
    image, a comma-joined list for several."""
    if len(expanded_list) == 1:
        return ["expanded_prompt: {}".format(expanded_list[0])]
    return ["expanded_prompts: {}".format(", ".join(expanded_list))]


def _common_entries(width, height, prior_steps, decoder_steps,
                    guidance, prior_guidance):
    """The settings lines shared by every generation mode, in order."""
    return [
        "width: {}".format(int(width)),
        "height: {}".format(int(height)),
        "prior_steps: {}".format(int(prior_steps)),
        "decoder_steps: {}".format(int(decoder_steps)),
        "guidance_scale: {}".format(float(guidance)),
        "prior_guidance_scale: {}".format(float(prior_guidance)),
    ]


def _run_txt2img(prior, decoder, template, negative, width, height,
                 prior_steps, decoder_steps, guidance, prior_guidance,
                 seeds, folder, start=1):
    """One text-to-image run per seed. Each image gets its own fresh
    dynamic-prompt expansion and its own generator, exactly as the old
    single-image flow did, so each image's seed reproduces it.
    Frames are numbered from start (6-digit), so a repeat run in a
    reused folder continues where the last run left off.
    Returns (frame paths, expanded prompts actually used)."""
    import torch

    frame_paths = []
    expanded_list = []
    for i, actual_seed in enumerate(seeds):
        expanded = _expand_templates(template)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        prior_out = prior(
            prompt=expanded,
            negative_prompt=negative,
            num_inference_steps=int(prior_steps),
            guidance_scale=float(prior_guidance),
            generator=generator,
        )
        image = decoder(
            image_embeds=prior_out.image_embeds,
            negative_image_embeds=prior_out.negative_image_embeds,
            height=int(height),
            width=int(width),
            num_inference_steps=int(decoder_steps),
            guidance_scale=float(guidance),
            generator=generator,
        ).images[0]
        frame_path = os.path.join(folder, "{:06d}.png".format(start + i))
        image.save(frame_path)
        frame_paths.append(frame_path)
        expanded_list.append(expanded)
    return frame_paths, expanded_list


def _txt2img_entries(template, expanded_list, negative_prompt, width,
                     height, prior_steps, decoder_steps, guidance,
                     prior_guidance, seeds):
    """settings.txt entries for a text-to-image run, in the usual
    order. The expanded prompt line is what was actually used (the
    template line keeps what was typed)."""
    return (
        ["prompt: {}".format(template)]
        + _expanded_entries(expanded_list)
        + ["negative_prompt: {}".format(negative_prompt or "")]
        + _common_entries(width, height, prior_steps, decoder_steps,
                          guidance, prior_guidance)
        + [_seed_entry(seeds),
           "prior_model: {}".format(PRIOR_ID),
           "decoder_model: {}".format(DECODER_ID),
           "dtype: float16",
           "device: cuda"]
    )


def _history_add(history, prompt):
    """Prepend a prompt to the session history: most recent first,
    last 10, duplicates collapsed. Never written to disk."""
    history = history or []
    return [prompt] + [p for p in history if p != prompt][:9]


def generate(prompt, negative_prompt, queue_text, width, height,
             prior_steps, decoder_steps, guidance, prior_guidance,
             seed, batch_name, images_per_generate, last_folder,
             history):
    """Text-to-image generation. Returns (gallery image list,
    status text, history dropdown update, last batch folder,
    history list). The gallery shows the saved frames.

    If the prompt queue has any non-empty lines, the queue runs
    INSTEAD of the single prompt box: each line is its own run with
    the current settings, sharing the one batch folder, frames
    numbered continuously across lines, one settings block per line.
    Otherwise the single prompt box runs as before. Repeat runs
    with the same batch name reuse the folder and keep numbering
    upward, so nothing is ever overwritten."""
    import torch

    lines = [line.strip() for line in (queue_text or "").splitlines()
             if line.strip()]
    if not lines and (not prompt or not prompt.strip()):
        return [], "Type a prompt first.", gr_update(), last_folder, \
            history
    if not torch.cuda.is_available():
        return [], (
            "No CUDA GPU found. Kandinsky UI needs an NVIDIA GPU with "
            "a recent driver."
        ), gr_update(), last_folder, history
    try:
        prior, decoder = _load_pipes()
    except Exception as exc:  # noqa: BLE001 -- show it, don't crash the UI
        return [], "Model load failed: {}".format(exc), gr_update(), \
            last_folder, history

    # Images per Generate is clamped to 1-20: the box already limits
    # it, but a typed value can slip past, so the code enforces it.
    count = int(images_per_generate) if images_per_generate else 1
    count = max(1, min(20, count))
    negative = (negative_prompt or "").strip() or None
    history = history or []

    folder = _batch_folder(_safe_batch_name(batch_name))
    os.makedirs(folder, exist_ok=True)
    if lines:
        all_frames = []
        all_seeds = []
        first_frame = None
        for line in lines:
            seeds = _seeds_for(seed, count)
            start = _next_frame_number(folder)
            if first_frame is None:
                first_frame = start
            frames, expanded = _run_txt2img(
                prior, decoder, line, negative, width, height,
                prior_steps, decoder_steps, guidance, prior_guidance,
                seeds, folder, start)
            _append_settings(folder, _run_header(start, len(frames)),
                             _txt2img_entries(
                                 line, expanded, negative_prompt,
                                 width, height, prior_steps,
                                 decoder_steps, guidance,
                                 prior_guidance, seeds))
            all_frames.extend(frames)
            all_seeds.extend(seeds)
            history = _history_add(history, line)
        status = "Queue done: {} prompts, {} images. Saved to {} ({}, {})".format(
            len(lines), len(all_frames), folder,
            _frame_range_text(first_frame, len(all_frames)),
            _seed_text(all_seeds))
        return (all_frames, status, _history_update(history),
                folder, history)

    seeds = _seeds_for(seed, count)
    start = _next_frame_number(folder)
    frames, expanded = _run_txt2img(
        prior, decoder, prompt, negative, width, height,
        prior_steps, decoder_steps, guidance, prior_guidance,
        seeds, folder, start)
    _append_settings(folder, _run_header(start, len(frames)),
                     _txt2img_entries(
                         prompt, expanded, negative_prompt, width,
                         height, prior_steps, decoder_steps,
                         guidance, prior_guidance, seeds))
    status = "Saved to {} ({}, {})".format(
        folder, _frame_range_text(start, len(frames)),
        _seed_text(seeds))
    history = _history_add(history, prompt)
    return (frames, status, _history_update(history), folder, history)


def generate_blend(prompt, blend_prompt, mix, negative_prompt, width,
                   height, prior_steps, decoder_steps, guidance,
                   prior_guidance, seed, batch_name,
                   images_per_generate, last_folder, history):
    """Prompt blending: run the prior for the main prompt AND the
    blend prompt (both with the main negative prompt), linearly
    interpolate both embeddings by the mix weight (0 = main prompt
    only, 1 = blend prompt only), then run the decoder once per
    image on the blended embeddings. Returns the same five values
    as generate(), into the main gallery and status."""
    import torch

    if not prompt or not prompt.strip():
        return [], "Type a prompt first.", gr_update(), last_folder, \
            history
    if not blend_prompt or not blend_prompt.strip():
        return [], "Type a blend prompt first.", gr_update(), \
            last_folder, history
    if not torch.cuda.is_available():
        return [], (
            "No CUDA GPU found. Kandinsky UI needs an NVIDIA GPU with "
            "a recent driver."
        ), gr_update(), last_folder, history
    try:
        prior, decoder = _load_pipes()
    except Exception as exc:  # noqa: BLE001 -- show it, don't crash the UI
        return [], "Model load failed: {}".format(exc), gr_update(), \
            last_folder, history

    count = int(images_per_generate) if images_per_generate else 1
    count = max(1, min(20, count))
    seeds = _seeds_for(seed, count)
    negative = (negative_prompt or "").strip() or None
    weight = float(mix) if mix is not None else 0.5
    weight = max(0.0, min(1.0, weight))

    folder = _batch_folder(_safe_batch_name(batch_name))
    os.makedirs(folder, exist_ok=True)
    start = _next_frame_number(folder)
    frame_paths = []
    expanded_main = []
    expanded_blend = []
    for i, actual_seed in enumerate(seeds):
        # Fresh expansion per image, so every image blends a newly
        # picked pair of prompts.
        exp_main = _expand_templates(prompt)
        exp_blend = _expand_templates(blend_prompt)
        # One generator per image, shared by both prior runs and the
        # decoder, so the image's seed reproduces the whole blend.
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        out_main = prior(
            prompt=exp_main,
            negative_prompt=negative,
            num_inference_steps=int(prior_steps),
            guidance_scale=float(prior_guidance),
            generator=generator,
        )
        out_blend = prior(
            prompt=exp_blend,
            negative_prompt=negative,
            num_inference_steps=int(prior_steps),
            guidance_scale=float(prior_guidance),
            generator=generator,
        )
        blended = (1.0 - weight) * out_main.image_embeds \
            + weight * out_blend.image_embeds
        blended_neg = (1.0 - weight) * out_main.negative_image_embeds \
            + weight * out_blend.negative_image_embeds
        image = decoder(
            image_embeds=blended,
            negative_image_embeds=blended_neg,
            height=int(height),
            width=int(width),
            num_inference_steps=int(decoder_steps),
            guidance_scale=float(guidance),
            generator=generator,
        ).images[0]
        frame_path = os.path.join(folder, "{:06d}.png".format(start + i))
        image.save(frame_path)
        frame_paths.append(frame_path)
        expanded_main.append(exp_main)
        expanded_blend.append(exp_blend)

    if len(expanded_main) == 1:
        exp_main_entries = ["expanded_prompt: {}".format(expanded_main[0])]
        exp_blend_entries = [
            "expanded_blend_prompt: {}".format(expanded_blend[0])]
    else:
        exp_main_entries = ["expanded_prompts: {}".format(
            ", ".join(expanded_main))]
        exp_blend_entries = ["expanded_blend_prompts: {}".format(
            ", ".join(expanded_blend))]
    _append_settings(folder, _run_header(start, len(frame_paths)), (
        ["prompt: {}".format(prompt),
         "blend_prompt: {}".format(blend_prompt),
         "mix: {}".format(weight)]
        + exp_main_entries
        + exp_blend_entries
        + ["negative_prompt: {}".format(negative_prompt or "")]
        + _common_entries(width, height, prior_steps, decoder_steps,
                          guidance, prior_guidance)
        + [_seed_entry(seeds),
           "prior_model: {}".format(PRIOR_ID),
           "decoder_model: {}".format(DECODER_ID),
           "dtype: float16",
           "device: cuda"]
    ))
    status = "Saved to {} (blend {}, {}, {})".format(
        folder, weight, _frame_range_text(start, len(frame_paths)),
        _seed_text(seeds))
    history = _history_add(history or [], prompt)
    return (frame_paths, status, _history_update(history), folder,
            history)


def _mask_from_editor(editor_value, fallback_image):
    """Split an ImageEditor value into (base image, mask image).

    The editor's painted layer carries the mask in its alpha channel:
    any paint means repaint there. The mask is rebuilt as white
    where paint exists, black elsewhere, so the brush color can
    never fool it (a black brush still masks). Returns (None, None)
    when there is no base image, and (base, None) when nothing was
    painted. The base falls back to the plain upload when the editor
    has no background of its own."""
    from PIL import Image as _Image

    background = None
    layers = []
    if isinstance(editor_value, dict):
        background = editor_value.get("background")
        layers = editor_value.get("layers") or []
    base = background if background is not None else fallback_image
    if base is None:
        return None, None
    if isinstance(base, _Image.Image):
        base = base.convert("RGB")
    else:
        try:
            base = _Image.open(base).convert("RGB")
        except Exception:  # noqa: BLE001 -- unreadable upload
            return None, None
    mask = None
    for layer in layers:
        if layer is None:
            continue
        try:
            rgba = layer.convert("RGBA")
        except Exception:  # noqa: BLE001 -- odd layer payload
            continue
        alpha = rgba.split()[3]
        candidate = alpha.point(lambda a: 255 if a > 0 else 0)
        extrema = candidate.getextrema()
        if extrema and extrema[1] > 0:
            mask = candidate.convert("L")
            break
    return base, mask


def _hint_from_image(pil_image, width, height):
    """Turn a user-supplied depth map into the ControlNet hint tensor,
    following Diffusers 0.39.0's own docstring example: RGB, resized
    to the generation size, float in [0, 1], channels first, one
    batch, fp16, on CUDA."""
    import numpy as np
    import torch
    from PIL import Image as _Image

    img = pil_image.convert("RGB").resize(
        (int(width), int(height)), resample=_Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    hint = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return hint.half().to("cuda")


def generate_tools(prompt, negative_prompt, width, height, prior_steps,
                   decoder_steps, guidance, prior_guidance, seed,
                   batch_name, images_per_generate, tool_image, mode,
                   strength, mask_editor, last_folder, history):
    """Image-tools generation. Uses the MAIN tab's controls for
    everything except the input image, mode, strength, and mask.
    Returns (tools gallery, tools status, history dropdown update,
    last batch folder, history list). The prompt queue does NOT
    apply here -- tools always use the single prompt box."""
    import torch

    if not prompt or not prompt.strip():
        return [], "Type a prompt first.", gr_update(), last_folder, \
            history
    if not torch.cuda.is_available():
        return [], (
            "No CUDA GPU found. Kandinsky UI needs an NVIDIA GPU with "
            "a recent driver."
        ), gr_update(), last_folder, history
    mode = mode or "Image to Image"
    if mode == "Image to Image" and tool_image is None:
        return [], "Upload an input image first.", gr_update(), \
            last_folder, history
    if mode == "ControlNet depth" and tool_image is None:
        return [], "Upload a depth map first.", gr_update(), \
            last_folder, history
    if mode == "Inpaint":
        base_image, mask_image = _mask_from_editor(mask_editor,
                                                   tool_image)
        if base_image is None:
            return [], "Upload an image first.", gr_update(), \
                last_folder, history
        if mask_image is None:
            return [], "Paint the area to repaint in the mask editor "
            "first.", gr_update(), last_folder, history
    try:
        prior, decoder = _load_pipes()
        if mode == "Image to Image":
            mode_pipe = _load_img2img(decoder)
        elif mode == "Inpaint":
            mode_pipe = _load_inpaint()
        else:
            mode_pipe = _load_controlnet()
    except Exception as exc:  # noqa: BLE001 -- show it, don't crash the UI
        return [], "Model load failed: {}".format(exc), gr_update(), \
            last_folder, history

    count = int(images_per_generate) if images_per_generate else 1
    count = max(1, min(20, count))
    seeds = _seeds_for(seed, count)
    negative = (negative_prompt or "").strip() or None
    strength_value = float(strength) if strength is not None else 0.5
    strength_value = max(0.0, min(1.0, strength_value))
    hint = None
    if mode == "ControlNet depth":
        hint = _hint_from_image(tool_image, width, height)

    folder = _batch_folder(_safe_batch_name(batch_name))
    os.makedirs(folder, exist_ok=True)
    start = _next_frame_number(folder)
    # Keep the input (and the mask, for inpaint) next to the frames
    # so a batch can always be reproduced from its folder alone.
    # Suffixed with the run's first frame number, so repeat runs
    # in a reused folder never collide on these files.
    if mode == "Inpaint":
        base_image.save(os.path.join(
            folder, "input_{:06d}.png".format(start)))
        mask_image.save(os.path.join(
            folder, "mask_{:06d}.png".format(start)))
    else:
        tool_image.convert("RGB").save(os.path.join(
            folder, "input_{:06d}.png".format(start)))

    frame_paths = []
    expanded_list = []
    for i, actual_seed in enumerate(seeds):
        expanded = _expand_templates(prompt)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        prior_out = prior(
            prompt=expanded,
            negative_prompt=negative,
            num_inference_steps=int(prior_steps),
            guidance_scale=float(prior_guidance),
            generator=generator,
        )
        if mode == "Image to Image":
            image = mode_pipe(
                image_embeds=prior_out.image_embeds,
                image=tool_image,
                negative_image_embeds=prior_out.negative_image_embeds,
                height=int(height),
                width=int(width),
                num_inference_steps=int(decoder_steps),
                guidance_scale=float(guidance),
                strength=strength_value,
                generator=generator,
            ).images[0]
        elif mode == "Inpaint":
            image = mode_pipe(
                image_embeds=prior_out.image_embeds,
                image=base_image,
                mask_image=mask_image,
                negative_image_embeds=prior_out.negative_image_embeds,
                height=int(height),
                width=int(width),
                num_inference_steps=int(decoder_steps),
                guidance_scale=float(guidance),
                generator=generator,
            ).images[0]
        else:
            image = mode_pipe(
                image_embeds=prior_out.image_embeds,
                negative_image_embeds=prior_out.negative_image_embeds,
                hint=hint,
                height=int(height),
                width=int(width),
                num_inference_steps=int(decoder_steps),
                guidance_scale=float(guidance),
                generator=generator,
            ).images[0]
        frame_path = os.path.join(folder, "{:06d}.png".format(start + i))
        image.save(frame_path)
        frame_paths.append(frame_path)
        expanded_list.append(expanded)

    if mode == "Image to Image":
        model_line = "decoder_model: {}".format(DECODER_ID)
        extra = ["strength: {}".format(strength_value)]
    elif mode == "Inpaint":
        model_line = "inpaint_model: {}".format(INPAINT_ID)
        extra = []
    else:
        model_line = "controlnet_model: {}".format(CONTROLNET_ID)
        extra = []
    _append_settings(folder, _run_header(start, len(frame_paths)), (
        ["mode: {}".format(mode),
         "prompt: {}".format(prompt)]
        + _expanded_entries(expanded_list)
        + ["negative_prompt: {}".format(negative_prompt or "")]
        + _common_entries(width, height, prior_steps, decoder_steps,
                          guidance, prior_guidance)
        + extra
        + [_seed_entry(seeds),
           "prior_model: {}".format(PRIOR_ID),
           model_line,
           "dtype: float16",
           "device: cuda"]
    ))
    status = "Saved to {} ({}, {}, {})".format(
        folder, mode, _frame_range_text(start, len(frame_paths)),
        _seed_text(seeds))
    history = _history_add(history or [], prompt)
    return (frame_paths, status, _history_update(history), folder,
            history)


def random_prompt():
    """Assemble one prompt from the in-code word banks. No list file,
    nothing to maintain -- just a starting point to edit or reroll."""
    return "A {} {}, {}, {}".format(
        random.choice(_RANDOM_SUBJECTS),
        random.choice(_RANDOM_PLACES),
        random.choice(_RANDOM_STYLES),
        random.choice(_RANDOM_LIGHTING),
    )


def star_batch(last_folder):
    """Copy the most recently generated batch folder into
    images_out/favorites/ under a unique name (never overwrite).
    If nothing has run yet this session, say so in the status."""
    if not last_folder or not os.path.isdir(last_folder):
        return "No batch has run yet in this session."
    os.makedirs(FAVORITES_DIR, exist_ok=True)
    base = os.path.basename(os.path.normpath(last_folder))
    dest = _unique_folder(base, root=FAVORITES_DIR)
    shutil.copytree(last_folder, dest)
    return "Starred to {}".format(dest)


def _history_update(choices):
    """Dropdown update helper, kept in one place so generate() stays
    readable. Gradio import is deferred to UI build time."""
    import gradio as gr
    return gr.update(choices=choices)


def gr_update():
    """No-change update for the history dropdown (early exits)."""
    import gradio as gr
    return gr.update()


def _on_mode_change(mode):
    """Show only the controls the chosen Image tools mode uses, and
    label the upload for what that mode expects. Strength belongs
    to img2img, the mask editor to inpaint, and ControlNet wants a
    depth map, not a photo."""
    import gradio as gr

    if mode == "Inpaint":
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(label="Input image"))
    if mode == "ControlNet depth":
        return (gr.update(visible=False),
                gr.update(visible=False),
                gr.update(label="Control image (depth map)"))
    return (gr.update(visible=True),
            gr.update(visible=False),
            gr.update(label="Input image"))


def _sync_editor(tool_image):
    """Mirror the plain upload into the mask editor's background, so
    painting a mask never needs a second upload. Changing the input
    resets the mask, since a mask belongs to its image."""
    if tool_image is None:
        return None
    return {"background": tool_image, "layers": [], "composite": None}


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Kandinsky 2.2") as demo:
        gr.Markdown(
            "# Kandinsky 2.2\n"
            "Text-to-image on your own GPU -- no Jupyter, no Colab. "
            "First Generate downloads the models (about 8-12 GB), so "
            "it takes a while; after that, repeat runs are just the render."
        )
        # Session-only state, shared by both tabs: never written
        # to disk.
        last_folder_state = gr.State(None)
        history_state = gr.State([])

        with gr.Tab("Text to image"):
            prompt = gr.Textbox(label="Prompt", lines=3,
                                value=DEFAULT_PROMPT)
            negative_prompt = gr.Textbox(label="Negative prompt",
                                         lines=2, value=DEFAULT_NEGATIVE)
            queue_text = gr.Textbox(
                label="Prompt queue (one prompt per line)", lines=4)
            gr.Markdown(
                "When the queue has lines, Generate runs the queue "
                "instead of the prompt above -- one run per line, "
                "all sharing the batch folder, frames numbered "
                "continuously, one settings block per line."
            )
            with gr.Row():
                random_btn = gr.Button("Random prompt")
            with gr.Row():
                width = gr.Slider(512, 1024, step=64, value=768,
                                  label="Width")
                height = gr.Slider(512, 1024, step=64, value=768,
                                   label="Height")
            with gr.Row():
                square_btn = gr.Button("Square")
                portrait_btn = gr.Button("Portrait")
                landscape_btn = gr.Button("Landscape")
            with gr.Row():
                prior_steps = gr.Slider(10, 100, step=1, value=25,
                                        label="Prior steps")
                decoder_steps = gr.Slider(10, 150, step=1, value=50,
                                          label="Decoder steps")
            with gr.Row():
                # Prior guidance defaults to 4.0, same as Guidance,
                # so a default run behaves exactly as it did when
                # one control drove both stages.
                prior_guidance = gr.Slider(1.0, 10.0, step=0.5,
                                           value=4.0,
                                           label="Prior guidance")
                guidance = gr.Slider(1.0, 10.0, step=0.5, value=4.0,
                                     label="Guidance scale")
                seed = gr.Number(value=-1, precision=0,
                                 label="Seed (-1 = random)")
            batch_name = gr.Textbox(label="Batch name",
                                    value="kandinsky-test")
            images_per_generate = gr.Number(
                value=1, precision=0, minimum=1, maximum=20,
                label="Images per Generate")
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop")
                star_btn = gr.Button("Star batch")
            with gr.Accordion("Prompt blending", open=False):
                gr.Markdown(
                    "Run the prior for both prompts, then mix their "
                    "embeddings: 0 is the main prompt only, 1 is the "
                    "blend prompt only. Dynamic {a|b|c} groups work "
                    "in both boxes."
                )
                blend_prompt = gr.Textbox(label="Blend prompt",
                                          lines=3)
                mix = gr.Slider(0.0, 1.0, step=0.05, value=0.5,
                                label="Mix")
                blend_btn = gr.Button("Blend")
            gallery = gr.Gallery(label="Results")
            status = gr.Textbox(label="Status")
            history = gr.Dropdown(label="Prompt history", choices=[],
                                  interactive=True)

        with gr.Tab("Image tools"):
            gr.Markdown(
                "The prompt, negative prompt, size, steps, guidance, "
                "seed, batch name, and images-per-click come from "
                "the Text to image tab -- everything here only adds "
                "the input image and the mode.\n\n"
                "- Image to Image: reshape the input picture toward "
                "the prompt. Strength 0 keeps it, 1 replaces it.\n"
                "- Inpaint: upload a picture, paint over the part "
                "to change, and only that part is regenerated.\n"
                "- ControlNet depth: the input must be a depth map "
                "(white near, black far), not a photo -- no depth "
                "estimation runs in this version."
            )
            tool_image = gr.Image(type="pil", label="Input image")
            mode = gr.Dropdown(
                label="Mode",
                choices=["Image to Image", "Inpaint",
                         "ControlNet depth"],
                value="Image to Image")
            strength = gr.Slider(0.0, 1.0, step=0.05, value=0.5,
                                 label="Strength (img2img only)")
            mask_editor = gr.ImageEditor(
                type="pil", label="Paint mask (Inpaint mode)",
                visible=False)
            with gr.Row():
                tools_generate_btn = gr.Button("Generate",
                                               variant="primary")
                tools_stop_btn = gr.Button("Stop")
            tools_gallery = gr.Gallery(label="Results")
            tools_status = gr.Textbox(label="Status")

        random_btn.click(random_prompt, outputs=[prompt])
        square_btn.click(lambda: (768, 768), outputs=[width, height])
        portrait_btn.click(lambda: (768, 1024),
                           outputs=[width, height])
        landscape_btn.click(lambda: (1024, 768),
                            outputs=[width, height])
        generate_event = generate_btn.click(
            generate,
            inputs=[prompt, negative_prompt, queue_text, width,
                    height, prior_steps, decoder_steps, guidance,
                    prior_guidance, seed, batch_name,
                    images_per_generate, last_folder_state,
                    history_state],
            outputs=[gallery, status, history, last_folder_state,
                     history_state],
        )
        blend_event = blend_btn.click(
            generate_blend,
            inputs=[prompt, blend_prompt, mix, negative_prompt,
                    width, height, prior_steps, decoder_steps,
                    guidance, prior_guidance, seed, batch_name,
                    images_per_generate, last_folder_state,
                    history_state],
            outputs=[gallery, status, history, last_folder_state,
                     history_state],
        )
        # Gradio 5 cancel pattern: Stop cancels the running Generate
        # and Blend events. Queued runs stop at once; a run already
        # diffusing is interrupted by the queue (verify on the PC).
        # Cancelling an event that is not running is a no-op.
        stop_btn.click(fn=None, inputs=[], outputs=[],
                       cancels=[generate_event, blend_event])
        star_btn.click(star_batch, inputs=[last_folder_state],
                       outputs=[status])
        history.change(fn=lambda choice: choice or "",
                       inputs=[history], outputs=[prompt])
        mode.change(fn=_on_mode_change, inputs=[mode],
                    outputs=[strength, mask_editor, tool_image])
        tool_image.change(fn=_sync_editor, inputs=[tool_image],
                          outputs=[mask_editor])
        tools_event = tools_generate_btn.click(
            generate_tools,
            inputs=[prompt, negative_prompt, width, height,
                    prior_steps, decoder_steps, guidance,
                    prior_guidance, seed, batch_name,
                    images_per_generate, tool_image, mode, strength,
                    mask_editor, last_folder_state, history_state],
            outputs=[tools_gallery, tools_status, history,
                     last_folder_state, history_state],
        )
        tools_stop_btn.click(fn=None, inputs=[], outputs=[],
                             cancels=[tools_event])
    return demo


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    build_ui().launch(server_name="127.0.0.1", inbrowser=True,
                      allowed_paths=[IMAGES_DIR])
