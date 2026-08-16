p = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/__init__.py"
s = open(p).read()
targets = {
 "    from .fp_quant import FPQuantConfig": ("FPQuantConfig",),
 "    from .humming import HummingConfig": ("HummingConfig",),
 "    from .inc import INCConfig": ("INCConfig",),
 "    from .torchao import TorchAOConfig": ("TorchAOConfig",),
 "    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig": ("QuarkConfig",),
 "    from vllm.models.deepseek_v4 import DeepseekV4FP8Config": ("DeepseekV4FP8Config",),
 "    from .mxfp4 import GptOssMxfp4Config, Mxfp4Config": ("GptOssMxfp4Config","Mxfp4Config"),
}
for line, names in targets.items():
    assigns = " = ".join(names) + " = None"
    repl = "    try:\n        " + line.strip() + "\n    except Exception:\n        " + assigns
    if repl in s:
        print("already guarded", names)
    elif line in s:
        s = s.replace(line, repl, 1); print("guarded", names)
    else:
        print("NOTFOUND", line.strip())
open(p,"w").write(s); print("quant backend imports patched")

# The guarded imports above intentionally leave unavailable backends as None.
# vLLM's model quantization auto-detection iterates every registered backend,
# so it must skip those sentinels before calling the config classmethod.
p = "/opt/venv/lib/python3.12/site-packages/vllm/config/model.py"
s = open(p).read()
needle = "                method = me_quant.get_quantization_config(name)"
replacement = needle + "\n                if method is None:\n                    continue"
if replacement not in s:
    if needle not in s:
        raise RuntimeError("vLLM quantization auto-detection hook not found")
    s = s.replace(needle, replacement, 1)
    open(p, "w").write(s)
    print("guarded unavailable quantization configs")
else:
    print("quantization config guard already present")
print("PATCH_DONE")
