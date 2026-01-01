"""
FlowMatch Scheduler Nodes for ComfyUI
Based on ai-toolkit's CustomFlowMatchEulerDiscreteScheduler
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import comfy.samplers
import comfy.model_management

# Try to import LogNormal, fallback if not available
try:
    from torch.distributions import LogNormal
    HAS_LOGNORMAL = True
except ImportError:
    HAS_LOGNORMAL = False
    print("Warning: LogNormal distribution not available. Lognorm blend scheduler will use fallback.")


class FlowMatchScheduler:
    """
    Generic FlowMatch scheduler compatible with various models including
    Flux, Qwen, Lumina, Z-Image-Turbo, HiDream, etc.
    Based on ai-toolkit's CustomFlowMatchEulerDiscreteScheduler
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 1000,
                    "tooltip": "Number of denoising steps"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Denoising strength (1.0 = full generation, <1.0 = img2img)"
                }),
                "scheduler_type": ([
                    "linear",
                    "sigmoid", 
                    "shift",
                    "shift_exponential",
                    "lognorm_blend",
                    "weighted",
                    "flux_shift"
                ], {
                    "tooltip": "Type of timestep distribution"
                }),
                "shift": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Manual shift value (overrides base_shift if != 1.0)"
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Base shift for dynamic shifting (lower = more consistent)"
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.16, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Max shift for dynamic shifting (higher = more variation)"
                }),
                "shift_terminal": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.001,
                    "tooltip": "Terminal value for sigma stretching (Qwen/Z-Image: 0.02)"
                }),
                "use_dynamic_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable resolution-based dynamic shifting"
                }),
                "time_shift_type": (["exponential", "linear"], {
                    "tooltip": "Type of time shift transformation"
                }),
                "invert_sigmas": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert sigma schedule (for models like Mochi)"
                }),
                "use_bell_curve": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply bell-shaped timestep weighting"
                }),
                "half_bell": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use half-bell curve (emphasis on early denoising)"
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Model (optional, for compatibility)"
                }),
                "latent_image": ("LATENT", {
                    "tooltip": "Latent image for dynamic shift calculation"
                }),
                "base_seq_len": ("INT", {
                    "default": 256, 
                    "min": 1, 
                    "max": 4096,
                    "tooltip": "Base sequence length for dynamic shifting"
                }),
                "max_seq_len": ("INT", {
                    "default": 4096, 
                    "min": 1, 
                    "max": 16384,
                    "tooltip": "Maximum sequence length for dynamic shifting"
                }),
                "patch_size": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 8,
                    "tooltip": "Patch size for transformer models"
                }),
                "lognorm_alpha": ("FLOAT", {
                    "default": 0.75, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Lognormal blend ratio (alpha)"
                }),
                "lognorm_scale": ("FLOAT", {
                    "default": 0.333, 
                    "min": 0.1, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Lognormal distribution scale"
                }),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_schedulers/flowmatch"
    DESCRIPTION = "Advanced FlowMatch scheduler with support for various models and timestep distributions"

    def calculate_shift(self, image_seq_len: int, base_seq_len: int, 
                       max_seq_len: int, base_shift: float, max_shift: float) -> float:
        """Calculate dynamic shift based on image sequence length (Flux/Qwen style)"""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor, 
                  shift_type: str = "exponential") -> torch.Tensor:
        """Apply time shift transformation"""
        if shift_type == "exponential":
            # Exponential shift (Flux/Qwen/Lumina style)
            shifted = []
            for timestep in t:
                if timestep > 0:
                    s = torch.exp(torch.tensor(mu)) / (torch.exp(torch.tensor(mu)) + (1 / timestep - 1) ** sigma)
                    shifted.append(s)
                else:
                    shifted.append(timestep)
            return torch.stack(shifted) if len(shifted) > 0 else t
        else:
            # Linear shift: shift * sigma / (1 + (shift - 1) * sigma)
            return mu * t / (1 + (mu - 1) * t)

    def stretch_to_terminal(self, sigmas: torch.Tensor, terminal: float) -> torch.Tensor:
        """Stretch sigmas to reach terminal value"""
        if terminal > 0 and len(sigmas) > 1:
            # Ensure final non-zero sigma reaches terminal value
            scale = (1.0 - terminal) / (sigmas[0] - sigmas[-2]) if sigmas[0] > sigmas[-2] else 1.0
            stretched = terminal + (sigmas[:-1] - sigmas[-2]) * scale
            # Keep the final zero
            return torch.cat([stretched, sigmas[-1:]])
        return sigmas

    def apply_bell_curve_weight(self, sigmas: torch.Tensor, num_steps: int, 
                               half_bell: bool = False) -> torch.Tensor:
        """Apply bell-shaped mean-normalized timestep weighting (ai-toolkit style)"""
        device = sigmas.device
        x = torch.arange(num_steps, dtype=torch.float32, device=device)
        y = torch.exp(-2 * ((x - num_steps / 2) / num_steps) ** 2)
        
        # Shift minimum to 0
        y_shifted = y - y.min()
        
        # Scale to make mean 1
        weights = y_shifted * (num_steps / y_shifted.sum())
        
        if half_bell:
            # Flatten second half to max (for emphasis on early denoising)
            weights[num_steps // 2:] = weights[num_steps // 2:].max()
        
        # Apply weights to sigmas (excluding final zero)
        if len(sigmas) > num_steps:
            weighted_sigmas = sigmas[:-1] * (weights / weights.mean())
            return torch.cat([weighted_sigmas, sigmas[-1:]])
        else:
            return sigmas * (weights / weights.mean())

    def get_sigmas(self, steps: int, denoise: float = 1.0, scheduler_type: str = "linear",
                   shift: float = 1.0, base_shift: float = 0.5, max_shift: float = 1.16, 
                   shift_terminal: float = 0.0, use_dynamic_shift: bool = False,
                   time_shift_type: str = "exponential", invert_sigmas: bool = False,
                   use_bell_curve: bool = False, half_bell: bool = False,
                   model: Optional[Any] = None, latent_image: Optional[Dict] = None,
                   base_seq_len: int = 256, max_seq_len: int = 4096, patch_size: int = 2,
                   lognorm_alpha: float = 0.75, lognorm_scale: float = 0.333) -> Tuple[torch.Tensor]:
        
        device = comfy.model_management.get_torch_device()
        
        # Calculate actual steps based on denoise
        num_steps = int(steps * denoise) if denoise < 1.0 else steps
        if num_steps < 1:
            num_steps = 1
        
        # Generate base timesteps/sigmas based on scheduler type
        if scheduler_type == "linear":
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]
            
        elif scheduler_type == "sigmoid":
            # Sigmoid distribution - concentrated in center (ai-toolkit style)
            t = torch.sigmoid(torch.randn((num_steps,), device=device))
            sigmas = 1.0 - t
            sigmas, _ = torch.sort(sigmas, descending=True)
            
        elif scheduler_type in ["shift", "shift_exponential", "flux_shift"]:
            # Generate linear timesteps first
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]
            
            # Calculate shift value (mu)
            if use_dynamic_shift and latent_image is not None:
                # Dynamic shift based on image resolution
                latent = latent_image["samples"]
                h, w = latent.shape[2], latent.shape[3]
                
                # For Flux, double the patch size to simulate latent reduction
                if scheduler_type == "flux_shift":
                    patch_size = patch_size * 2
                
                # Calculate sequence length for transformer models
                image_seq_len = (h * w) // (patch_size ** 2)
                mu = self.calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift)
            else:
                # Static shift
                mu = shift if shift != 1.0 else base_shift
            
            # Apply time shift
            if scheduler_type == "shift_exponential" or time_shift_type == "exponential":
                sigmas = self.time_shift(mu, 1.0, sigmas, "exponential")
            else:
                sigmas = self.time_shift(mu, 1.0, sigmas, "linear")
                
        elif scheduler_type == "lognorm_blend":
            if HAS_LOGNORMAL:
                # Lognormal + linear blend (ai-toolkit style)
                lognormal = LogNormal(loc=0, scale=lognorm_scale)
                
                t1_count = int(num_steps * lognorm_alpha)
                t2_count = num_steps - t1_count
                
                if t1_count > 0:
                    t1 = lognormal.sample((t1_count,)).to(device)
                    t1 = 1.0 - (t1 / t1.max())
                else:
                    t1 = torch.tensor([], device=device)
                
                if t2_count > 0:
                    t2 = torch.linspace(1.0, 0.0, t2_count + 1, device=device)[:-1]
                else:
                    t2 = torch.tensor([], device=device)
                
                sigmas = torch.cat([t1, t2])
                sigmas, _ = torch.sort(sigmas, descending=True)
            else:
                # Fallback to sigmoid if LogNormal not available
                print("LogNormal not available, using sigmoid distribution instead")
                t = torch.sigmoid(torch.randn((num_steps,), device=device))
                sigmas = 1.0 - t
                sigmas, _ = torch.sort(sigmas, descending=True)
            
        elif scheduler_type == "weighted":
            # Start with linear
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]
            # Bell curve weighting will be applied later if use_bell_curve is True
        
        # Apply bell curve weighting if requested
        if use_bell_curve:
            sigmas = self.apply_bell_curve_weight(sigmas, num_steps, half_bell)
        
        # Apply terminal stretching if specified
        if shift_terminal > 0:
            sigmas = self.stretch_to_terminal(sigmas, shift_terminal)
        
        # Invert sigmas if requested (for certain models like Mochi)
        if invert_sigmas:
            sigmas = 1.0 - sigmas
        
        # Ensure descending order (unless inverted)
        if not invert_sigmas:
            sigmas, _ = torch.sort(sigmas, descending=True)
        else:
            sigmas, _ = torch.sort(sigmas, descending=False)
        
        # Add final sigma (0 for normal, 1 for inverted)
        final_sigma = torch.ones(1, device=device) if invert_sigmas else torch.zeros(1, device=device)
        sigmas = torch.cat([sigmas, final_sigma])
        
        return (sigmas,)


class FlowMatchSchedulerPresets:
    """Preset configurations for common models matching ai-toolkit configs"""
    
    PRESETS = {
        "flux_dev": {
            "scheduler_type": "flux_shift",
            "shift": 1.0,
            "base_shift": 0.5,
            "max_shift": 1.16,
            "shift_terminal": 0.0,
            "use_dynamic_shift": True,
            "time_shift_type": "linear",
            "base_seq_len": 256,
            "max_seq_len": 4096,
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "flux_schnell": {
            "scheduler_type": "flux_shift",
            "shift": 1.0,
            "base_shift": 0.5,
            "max_shift": 1.16,
            "shift_terminal": 0.0,
            "use_dynamic_shift": True,
            "time_shift_type": "linear",
            "base_seq_len": 256,
            "max_seq_len": 4096,
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "qwen_image": {
            "scheduler_type": "shift_exponential",
            "shift": 1.0,
            "base_shift": 0.5,
            "max_shift": 0.9,
            "shift_terminal": 0.02,
            "use_dynamic_shift": True,
            "time_shift_type": "exponential",
            "base_seq_len": 256,
            "max_seq_len": 8192,
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "z_image_turbo": {
            "scheduler_type": "shift_exponential",
            "shift": 1.0,
            "base_shift": 0.5,
            "max_shift": 0.9,
            "shift_terminal": 0.02,
            "use_dynamic_shift": True,
            "time_shift_type": "exponential",
            "base_seq_len": 256,
            "max_seq_len": 8192,
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "lumina": {
            "scheduler_type": "shift_exponential",
            "shift": 1.0,
            "base_shift": 0.6,
            "max_shift": 1.2,
            "shift_terminal": 0.0,
            "use_dynamic_shift": True,
            "time_shift_type": "exponential",
            "base_seq_len": 256,
            "max_seq_len": 4096,
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "hidream": {
            "scheduler_type": "shift",
            "shift": 1.0,
            "base_shift": 0.5,
            "max_shift": 1.16,
            "shift_terminal": 0.0,
            "use_dynamic_shift": False,
            "time_shift_type": "linear",
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "stable_diffusion": {
            "scheduler_type": "linear",
            "shift": 1.0,
            "base_shift": 1.0,
            "max_shift": 1.0,
            "shift_terminal": 0.0,
            "use_dynamic_shift": False,
            "time_shift_type": "linear",
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "mochi": {
            "scheduler_type": "shift",
            "shift": 1.0,
            "base_shift": 0.5,
            "max_shift": 1.16,
            "shift_terminal": 0.0,
            "use_dynamic_shift": True,
            "time_shift_type": "linear",
            "invert_sigmas": True,
            "use_bell_curve": False,
        },
        "ai_toolkit_sigmoid": {
            "scheduler_type": "sigmoid",
            "shift": 1.0,
            "base_shift": 1.0,
            "max_shift": 1.0,
            "shift_terminal": 0.0,
            "use_dynamic_shift": False,
            "time_shift_type": "linear",
            "invert_sigmas": False,
            "use_bell_curve": False,
        },
        "ai_toolkit_weighted": {
            "scheduler_type": "weighted",
            "shift": 1.0,
            "base_shift": 1.0,
            "max_shift": 1.0,
            "shift_terminal": 0.0,
            "use_dynamic_shift": False,
            "time_shift_type": "linear",
            "invert_sigmas": False,
            "use_bell_curve": True,
            "half_bell": False,
        },
        "custom": {
            "scheduler_type": "linear",
            "shift": 1.0,
            "base_shift": 1.0,
            "max_shift": 1.0,
            "shift_terminal": 0.0,
            "use_dynamic_shift": False,
            "time_shift_type": "linear",
            "invert_sigmas": False,
            "use_bell_curve": False,
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {
                    "tooltip": "Select a preset configuration for your model"
                }),
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 1000,
                    "tooltip": "Number of denoising steps"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Denoising strength"
                }),
            },
            "optional": {
                "latent_image": ("LATENT", {
                    "tooltip": "Optional: Latent image for dynamic shift calculation"
                }),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_schedulers/flowmatch"
    DESCRIPTION = "Quick preset configurations for common models (Flux, Qwen, Z-Image-Turbo, etc.)"

    def get_sigmas(self, preset: str, steps: int, denoise: float, 
                   latent_image: Optional[Dict] = None) -> Tuple[torch.Tensor]:
        scheduler = FlowMatchScheduler()
        config = self.PRESETS.get(preset, self.PRESETS["custom"])
        
        return scheduler.get_sigmas(
            steps=steps,
            denoise=denoise,
            latent_image=latent_image,
            **config
        )


class FlowMatchAutoConfig:
    """Automatically configure all sampling parameters for a model"""
    
    CONFIGS = {
        "flux_dev": {
            "steps": 28,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "flux_shift",
            "denoise": 1.0,
        },
        "flux_schnell": {
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "flux_shift", 
            "denoise": 1.0,
        },
        "qwen_image": {
            "steps": 20,
            "cfg": 2.5,
            "sampler_name": "euler",
            "scheduler": "shift_exponential",
            "denoise": 1.0,
        },
        "z_image_turbo": {
            "steps": 25,
            "cfg": 2.5,
            "sampler_name": "euler",
            "scheduler": "shift_exponential",
            "denoise": 1.0,
        },
        "lumina": {
            "steps": 25,
            "cfg": 4.0,
            "sampler_name": "euler",
            "scheduler": "shift_exponential",
            "denoise": 1.0,
        },
        "hidream": {
            "steps": 25,
            "cfg": 4.0,
            "sampler_name": "euler",
            "scheduler": "shift",
            "denoise": 1.0,
        },
        "stable_diffusion": {
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "linear",
            "denoise": 1.0,
        },
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_preset": (list(cls.CONFIGS.keys()), {
                    "tooltip": "Select your model type for automatic configuration"
                }),
            },
            "optional": {
                "override_steps": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 1000,
                    "tooltip": "Override default steps (0 = use preset)"
                }),
                "override_cfg": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 30.0,
                    "tooltip": "Override default CFG (0 = use preset)"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("steps", "cfg", "denoise", "sampler_name", "scheduler_type")
    FUNCTION = "get_config"
    CATEGORY = "sampling/custom_schedulers/flowmatch"
    DESCRIPTION = "Automatically outputs optimal sampling parameters for your model"
    
    def get_config(self, model_preset: str, override_steps: int = 0, 
                  override_cfg: float = 0.0) -> Tuple[int, float, float, str, str]:
        config = self.CONFIGS.get(model_preset, self.CONFIGS["stable_diffusion"])
        
        steps = override_steps if override_steps > 0 else config["steps"]
        cfg = override_cfg if override_cfg > 0 else config["cfg"]
        
        return (
            steps,
            cfg,
            config["denoise"],
            config["sampler_name"],
            config["scheduler"],
        )


class FlowMatchGuide:
    """Display recommended settings for each model"""
    
    RECOMMENDATIONS = {
        "flux_dev": """
FLUX Dev Recommended Settings:
- Steps: 20-50 (28 optimal)
- CFG: 3.5-7.0
- Sampler: euler
- Scheduler: flux_shift with dynamic shifting
- Notes: Higher resolution benefits from more steps
""",
        "flux_schnell": """
FLUX Schnell Recommended Settings:
- Steps: 4-8 (4 optimal for speed)
- CFG: 1.0 (no guidance)
- Sampler: euler
- Scheduler: flux_shift with dynamic shifting
- Notes: Optimized for speed, minimal steps needed
""",
        "qwen_image": """
Qwen Image Edit Recommended Settings:
- Steps: 20
- CFG: 2.5
- Sampler: euler
- Scheduler: shift_exponential
- Terminal: 0.02
- Notes: Exponential shift with terminal stretching
""",
        "z_image_turbo": """
Z-Image-Turbo Recommended Settings:
- Steps: 20-30 (25 optimal)
- CFG: 2.0-3.0
- Sampler: euler
- Scheduler: shift_exponential
- Terminal: 0.02
- Notes: Similar to Qwen, distilled for efficiency
""",
        "lumina": """
Lumina Recommended Settings:
- Steps: 20-30
- CFG: 3.0-5.0
- Sampler: euler
- Scheduler: shift_exponential
- Base shift: 0.6, Max shift: 1.2
- Notes: Wider shift range for varied generation
""",
        "hidream": """
HiDream Recommended Settings:
- Steps: 20-30
- CFG: 3.5-7.0
- Sampler: euler (flowmatch in ai-toolkit)
- Scheduler: shift (linear)
- Notes: Standard shift without dynamic adjustment
""",
        "stable_diffusion": """
Stable Diffusion Recommended Settings:
- Steps: 20-50
- CFG: 7.0
- Sampler: euler_ancestral
- Scheduler: linear
- Notes: Traditional linear schedule, no shift needed
""",
        "mochi": """
Mochi Recommended Settings:
- Steps: 20-30
- CFG: 3.5
- Sampler: euler
- Scheduler: shift with inverted sigmas
- Notes: Requires inverted sigma schedule
""",
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": (list(cls.RECOMMENDATIONS.keys()), {
                    "tooltip": "Select a model to see recommended settings"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("recommendations",)
    FUNCTION = "get_recommendation"
    CATEGORY = "sampling/custom_schedulers/flowmatch"
    DESCRIPTION = "Display detailed recommendations for each model type"
    
    def get_recommendation(self, preset: str) -> Tuple[str]:
        return (self.RECOMMENDATIONS.get(preset, "No recommendation available"),)