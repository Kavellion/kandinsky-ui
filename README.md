# Kandinsky UI
Easiest install: grab the bundled zip from the Releases tab — no Python setup needed.

**Kandinsky 2.2, without Jupyter and without Colab.**

Kandinsky 2.2 is a text-to-image model with a painterly, high-detail
look of its own -- a different engine from Disco Diffusion, not a
replacement for it. This is a plain app: run it, a page opens in your
browser, type a prompt, pictures appear. Built the same way as Disco
UI, free forever.

![Kandinsky UI with a generated image](https://github.com/user-attachments/assets/725aa732-4f06-47eb-9723-8914aff72c7f)

*Kandinsky UI with a generated image*

![Golden cyborg portrait made with Kandinsky UI](https://github.com/user-attachments/assets/bc4680fa-b2d6-43cc-a200-4d8396f564db)

*Example image made with the app: a golden cyborg portrait*

![Sailing ship in a sunlit library made with Kandinsky UI](https://github.com/user-attachments/assets/4c5ebb94-2819-4dd9-b3dd-df383ec017c6)

*Example image made with the app: a sailing ship in a sunlit library*

## What it does

- Text-to-image generation on your own GPU -- no cloud, no account
- Prompt + negative prompt, width/height, prior steps, decoder steps,
  guidance scale, prior guidance scale, seed, batch name
- Up to 20 pictures per Generate, each with its own seed, saved as
  000001.png, 000002.png, and so on
- Square / Portrait / Landscape size presets, a Random prompt button
  (word banks built in, no list file), and a session-only prompt
  history dropdown
- A Stop button to cancel a run, and Star batch to copy a batch into
  images_out/favorites/
- A prompt queue (one prompt per line -- when it has lines, Generate
  runs the queue instead of the single prompt box, all lines sharing
  the one batch folder, frames numbered continuously)
- Dynamic prompts: {a|b|c} groups, one random pick per group, in the
  prompt box, every queue line, and both blend prompts. settings.txt
  records the expanded prompt actually used, not the template
- Prompt blending: a second prompt box and a mix slider (0 = main
  prompt only, 1 = blend prompt only), with its own Blend button
- An Image tools tab: image to image (strength slider), inpainting
  (paint the area to change in the mask editor), and ControlNet depth
  (you supply the depth map -- no depth estimation runs here). The
  Image tools Generate reuses the main tab's prompt, size, steps,
  guidance, seed, batch name, and images-per-click
- Every run adds its pictures to the batch folder and appends a block
  to `settings.txt` next to them (date, time, frame range, settings)

No animation, no upscaling.

## Requirements

- Windows, an NVIDIA GPU (12 GB recommended), and a recent driver
- Python: nothing to install -- Python 3.12 is included in this package, bundled in `python_embeded\`. Any Python you already have is left completely alone
- Disk space: about 20 GB free. The app itself is small; the prior +
  decoder models (about 8-12 GB total) download on first Generate into
  the Hugging Face cache, and your pictures live in `images_out/`

## Install

1. Download this folder and unzip it, or `git clone https://github.com/Kavellion/kandinsky-ui.git`. Python is included -- there is nothing to install first.
2. Double-click `install.bat` and wait -- it sets up pip in the bundled Python and installs PyTorch with CUDA plus the packages. Run it again if anything interrupts it; it resumes where it stopped.
3. Double-click `run.bat`. A page opens in your browser. Type a prompt and hit Generate.

## First test

1. Leave everything at the defaults and hit Generate: 768x768,
   prior 25 steps, decoder 50 steps, guidance 4.0.
2. The first run downloads the models (about 8-12 GB), so it takes a
   while -- that is the one-time download, not a stuck render. After
   that, repeat runs are just the render.
3. When the 768 test works, try 1024x1024 (both sliders to 1024).
   If 1024 runs out of memory, drop back to 768 -- that size is the
   proven one.

## Where pictures go

`images_out/` next to `run.bat`, one subfolder per batch name, reused
across runs -- repeat runs with the same batch name add pictures to
the same folder, numbered continuously (`000001.png`, `000002.png`,
...), so nothing is ever overwritten. `settings.txt` in the folder
grows by one block per run (date, time, frame range, prompt, negative
prompt, size, steps, guidance, prior guidance, seed(s), and model
ids), so you can always see exactly what made a picture you like. For
dynamic prompts the expanded prompt actually used is recorded too.
Image-tools runs also save the input picture as
`input_<first frame>.png` (and the painted mask as
`mask_<first frame>.png` for inpainting). Starred batches land in
`images_out/favorites/`.

## How it works

Pipelines, all in fp16 on CUDA:

1. `KandinskyV22PriorPipeline` (`kandinsky-community/kandinsky-2-2-prior`)
   turns the prompt into CLIP image embeddings.
2. `KandinskyV22Pipeline` (`kandinsky-community/kandinsky-2-2-decoder`)
   turns those embeddings into the picture.
3. Image tools: `KandinskyV22Img2ImgPipeline` (built from the decoder's
   own components, so it costs no extra VRAM), `KandinskyV22InpaintPipeline`
   (`kandinsky-community/kandinsky-2-2-decoder-inpaint`), and
   `KandinskyV22ControlnetPipeline`
   (`kandinsky-community/kandinsky-2-2-controlnet-depth`).

Guidance here is Kandinsky's own scale (default 4.0). It has nothing
to do with Disco's 10000-scale clip guidance -- do not carry Disco
numbers over.

The inpaint and ControlNet weights download on first use of that mode,
like the main models do on first Generate. Each image-tools pipeline
stays loaded once used; if you run every mode in one session on a
12 GB card and hit out-of-memory, close the app and start it again
with `run.bat` to free VRAM. ControlNet takes a depth map you supply
(white near, black far) -- nothing estimates depth for you.

## Models and permissions

The models belong to their creators -- see their Hugging Face pages
for terms. Nothing is bundled here and nothing is sold. This UI (the
Gradio app, the wrapper, the installer) is by Alan Carmack.

## License

MIT for this UI code -- free to use, share, and change.
