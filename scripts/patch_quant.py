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
    if line in s:
        s = s.replace(line, repl, 1); print("guarded", names)
    else:
        print("NOTFOUND", line.strip())
open(p,"w").write(s); print("PATCH_DONE")
