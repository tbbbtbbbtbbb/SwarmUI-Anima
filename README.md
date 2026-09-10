# SwarmUI Anima

One reference image and one pose image for Anima, through SwarmUI's normal Generate
tab and API. Select your usual Anima preset, then add the controls you need.
Prompts, sampling, Turbo, and other LoRAs stay under your control. The extension
does not create or edit presets.

## Install

Requires current SwarmUI and ComfyUI with Anima support. The validated revisions
and generation checks are recorded in [TESTING.md](TESTING.md).

1. Clone into SwarmUI's extension directory:

   ```sh
   git clone https://github.com/tbbbtbbbtbbb/SwarmUI-Anima.git SwarmUI/src/Extensions/SwarmAnima
   ```

2. Restart SwarmUI. Its self-started ComfyUI backend automatically installs the
   pose dependency and downloads about **2.4 GB** of required model assets on its
   first start. Wait for the backend to finish loading. No pip commands, separate
   model downloads, or additional node plugins are needed.

First setup needs internet access and space for Character Reference 10, SigLIP2,
Pose Preview-2, and two detectors. Progress appears in **Server → Logs → ComfyUI**.
Downloads use pinned revisions and SHA-256 checks from
[models.json](ExtraNodes/SwarmAnimaNodes/models.json), are shared by backends using
the same model directory, and are reused on later starts. Existing files with
different contents are preserved and reported. An interrupted download is retried
on the next backend restart. Once setup succeeds, this extension works offline.

Setup installs `rtmlib` if missing or older than the supported version. Swarm
already supplies OpenCV and ONNX Runtime; plain ComfyUI installations get these
automatically if missing, with existing package versions constrained. Setup does
not upgrade PyTorch, CUDA packages, other installed libraries, or change presets. Use your
usual Anima checkpoint; the automatic downloads are this extension's control assets.

For an **external ComfyUI server** that Swarm does not manage, install this same
repository into that server's `ComfyUI/custom_nodes/SwarmAnima` and restart it.
The same automatic setup runs there. Installing on the Swarm host cannot install
code on a separate remote server. With a Swarm-to-Swarm
backend, install the extension in the remote Swarm instance too.

## Use

1. Select an Anima checkpoint or one of your existing Anima presets.
2. Drag or paste a reference into the **prompt box**, using Swarm's existing image
   prompt input. One reference guides appearance; extra images are ignored.
3. Open **Anima** and optionally upload or select a **Pose Image**. Use a clearly
   visible single person. The first detected person's pose is used.
4. Generate normally. Start with both strengths at `1`. Describe the intended
   character and scene; avoid a pose description that contradicts the pose image.

| Control | Default | Meaning |
|---|---:|---|
| Pose Image | empty | Detect and reuse this person's pose |
| Pose Strength | 1 | Scale pose conditioning; 0 bypasses the whole pose adapter |
| Reference Strength | 1 | Scale reference image attention; 0 bypasses the whole reference adapter |
| Primary Reference Image · advanced | 1 | Select one prompt image, counting from 1 |
| Reference Start · advanced | 0 | Begin image attention at this denoising fraction |
| Reference End · advanced | 1 | End image attention at this denoising fraction |

Start must be less than End. The timing controls gate the image-attention branch.
Companion LoRAs retain their trained scale while an adapter is enabled. Strength 0 or
removing the image bypasses the entire corresponding path, including preprocessing.
Anima controls are hidden and ignored for other model families.

Use the sampling settings recommended by your checkpoint or your own presets.
Turbo is an ordinary optional LoRA: this extension neither enables it nor changes
its weight, steps, CFG, or scheduler. 1024px is a useful starting point for pose
quality; lower resolutions are supported.

These are appearance and pose guidance models, not exact identity or pose locks.
Pose Preview-2 remains experimental, and fine-tunes or other LoRAs can weaken it.
If MiaoMiao misses the pose at strength `1`, try `1.3`; this helped in the recorded
tests. Keep adjusting the generation controls without editing your preset.
Multiple references do not provide character-slot binding or identity mixing.

## API

Use `GenerateText2Image` or `GenerateText2ImageWS`, with your usual session and
generation parameters. The new parameters use Swarm's normal lowercase IDs:

```json
{
  "session_id": "YOUR_SESSION",
  "presets": ["My Anima preset"],
  "prompt": "1girl, solo, adult woman, full body, blue jacket, gray background",
  "promptimages": ["data:image/png;base64,..."],
  "poseimage": "data:image/png;base64,...",
  "referencestrength": 1.0,
  "posestrength": 1.0,
  "primaryreferenceimage": 1,
  "referencestart": 0.0,
  "referenceend": 1.0
}
```

Reference and pose inputs are independent; omit either. `ListT2IParams` exposes
their metadata, and `ComfyGetGeneratedWorkflow` shows the resulting nodes. There
is no separate generation API.

## Cache

SigLIP2 features and detected keypoints/skeletons are stored as safetensors under
`ComfyUI/user/default/swarm_anima/cache/` (or your configured ComfyUI user directory).
They survive restarts and are shared by the extension's backend nodes. CPU
preprocessing runs only on a miss; encoder and detector instances are not retained.

The first use creates `user/default/swarm_anima/cache.json`:

```json
{
  "max_mb": 1024,
  "max_entries": 2000,
  "max_age_days": 0
}
```

Entries are evicted least recently used first. `max_mb: 0` disables disk caching;
zero entries/age limits mean unlimited entries/no age limit, within the size cap.
Keys include image pixels, preprocessing version, relevant library version, model
file identity, and pose output dimensions. File locking prevents duplicate misses
and atomic writes prevent partial entries. Corrupt entries are recomputed.

To inspect or clear the derived cache, stop ComfyUI and run:

```sh
python SwarmUI/src/Extensions/SwarmAnima/manage_cache.py --user-dir ComfyUI/user
python SwarmUI/src/Extensions/SwarmAnima/manage_cache.py --user-dir ComfyUI/user --clear
```

ComfyUI also caches node results in memory. Restart the backend after changing disk
cache limits when you need them to take effect immediately for an unchanged graph.

## Implementation and credits

The C# extension registers six parameters and adds nodes to Swarm's generated
workflow. Reference features use SigLIP2; pose uses DWPose WholeBody-133 and the
adapter's training skeleton style. Pose latents are normalized through Anima's
native latent format before control fusion. All trained adapter LoRAs use native
ComfyUI weight patches. Reference block overrides belong to cloned ModelPatchers
and are restored by ComfyUI when switching models or disabling the adapter.

- [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [LuciferTC's IP-Adapter implementation](https://github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter)
  and [Anima IP-Adapter weights](https://huggingface.co/LuciferTC/Anima-IP-Adapter)
- [Claquasse's Anima Control Pose](https://huggingface.co/Claquasse/Anima-Control-Pose)
- [Google SigLIP2](https://huggingface.co/google/siglip2-base-patch16-512)

Code: [Apache-2.0](LICENSE), with attribution in [NOTICE](NOTICE). Model weights
are downloaded separately and retain their original licenses, including Anima
and Pose Preview-2's non-commercial model terms. See the linked model cards.
