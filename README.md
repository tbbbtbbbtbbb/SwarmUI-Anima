# SwarmUI Anima

Reference images and pose control for Anima, using SwarmUI's normal Generate tab,
presets, metadata, and generation API. Works with Anima checkpoints and optional
Turbo LoRAs; sampling settings remain under your control.

Development in progress. Generation validation and installation instructions will
be added before release.

The integration uses Swarm's image prompt input for one selected reference and
adds a separate Pose Image. Derived reference features and detected poses are
cached on the backend. Model patches belong to a cloned ComfyUI ModelPatcher,
so disabling the controls does not leave changes on a loaded model.

Upstream references:

- https://github.com/mcmonkeyprojects/SwarmUI
- https://github.com/Comfy-Org/ComfyUI
- https://github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter
- https://huggingface.co/LuciferTC/Anima-IP-Adapter
- https://huggingface.co/Claquasse/Anima-Control-Pose
