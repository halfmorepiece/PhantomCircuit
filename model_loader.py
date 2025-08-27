import torch
import sys
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

def load_model(model_name, model_config, device_config=None):
    """
    Load the specified model and tokenizer.
    
    Args:
        model_name: model name
        model_config: model configuration dictionary
        device_config: device configuration dictionary (optional)
        
    Returns:
        tuple: (model, global_device) - HookedTransformer model instance and device
    """
    model_path = model_config["path"]
    dtype_str = model_config["dtype"] 
    trust_remote_code = model_config.get("trust_remote_code", False)

    # device selection
    def select_device(device_config):
        
        if device_config is None:
            device_config = {"force_device": None, "auto_fallback": True}
        
        force_device = device_config.get("force_device", None)
        auto_fallback = device_config.get("auto_fallback", True)
        
        # If a device is forced
        if force_device is not None and force_device != "auto":
            if force_device == "cpu":
                print(f"Forcing CPU device")
                return "cpu"
            elif force_device.startswith("cuda"):
                if torch.cuda.is_available():
                    try:
                        torch.cuda.set_device(force_device)
                        print(f"Forcing GPU device: {force_device}")
                        return force_device
                    except Exception as e:
                        if auto_fallback:
                            print(f"Specified GPU device {force_device} is not available ({e}), falling back to CPU")
                            return "cpu"
                        else:
                            raise RuntimeError(f"Specified GPU device {force_device} is not available: {e}")
                else:
                    if auto_fallback:
                        print(f"CUDA is not available, falling back to CPU")
                        return "cpu"
                    else:
                        raise RuntimeError(f"CUDA is not available, but GPU device {force_device} was specified")
        
        # Auto-detect device
        if torch.cuda.is_available():
            device = "cuda:0"
            try:
                torch.cuda.set_device(device)
                print(f"Auto-detection: Using GPU device {device}")
                return device
            except Exception as e:
                print(f"GPU device setup failed ({e}), using CPU")
                return "cpu"
        else:
            print("Auto-detection: Using CPU device")
            return "cpu"
    
    device = select_device(device_config)
    global_device = torch.device(device)

    hf_model = None
    tokenizer = None
    load_kwargs = {"trust_remote_code": trust_remote_code}
    final_compute_dtype = None

    try:
        # set load parameters based on dtype
        model_dtype = torch.float32
        if dtype_str == 'bfloat16':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
        elif dtype_str == 'float16':
            model_dtype = torch.float16
        load_kwargs["torch_dtype"] = model_dtype
        final_compute_dtype = model_dtype

        # load model
        hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )

        # try to enable gradient checkpointing
        gradient_checkpointing_enabled_flag = False
        if hasattr(hf_model, "gradient_checkpointing_enable"):
            try:
                hf_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if hasattr(hf_model, 'is_gradient_checkpointing') and hf_model.is_gradient_checkpointing:
                    gradient_checkpointing_enabled_flag = True
                elif hasattr(hf_model, 'config') and getattr(hf_model.config, 'use_cache', True) is False:
                    gradient_checkpointing_enabled_flag = True

            except TypeError:
                try:
                    hf_model.gradient_checkpointing_enable()
                    gradient_checkpointing_enabled_flag = True
                except Exception as e_gc_legacy:
                    pass
            except Exception as e_gc:
                pass
        else:
            pass

        # determine original model name for config
        original_model_name_for_config = model_name
        if model_name.endswith("-reinitialized"):
            original_model_name_for_config = model_name.replace("-reinitialized", "")

        hooked_transformer_dtype = final_compute_dtype

        with torch.no_grad():
            model = HookedTransformer.from_pretrained(
                model_name=original_model_name_for_config,
                hf_model=hf_model,
                tokenizer=tokenizer,
                device=global_device,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                torch_dtype=hooked_transformer_dtype
            )
            # ensure model is on the correct device
            model.to(global_device)
            model.eval()
            # Hook points configuration
            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True

        return model, global_device

    except ImportError as e_dep:
        print(f"Import error during model loading (possibly missing dependencies like bitsandbytes): {e_dep}")
        sys.exit(1)
    except OSError as e:
        print(f"OS error when loading model from '{model_path}': {e}")
        print("Ensure the path is correct and contains the model files (config.json, pytorch_model.bin etc).")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error when loading model: {e}")
        traceback.print_exc()
        sys.exit(1) 