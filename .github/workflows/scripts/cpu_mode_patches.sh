#!/bin/bash
# CPU Mode Patches Script
# This script creates mock implementations of hardware-dependent modules
# to ensure tests can run in CPU-only mode in CI environments.

echo "Creating CPU mode patches..."

# Create sitecustomize.py that forces CPU mode for ML frameworks
cat > sitecustomize.py << 'EOF'
import os
import sys

# Force CPU-only mode for various ML frameworks
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['XPU_VISIBLE_DEVICES'] = '-1'
os.environ['FORCE_CPU_ONLY'] = '1'

# Patch torch to ensure it's in CPU-only mode
def patch_torch():
    try:
        import torch
        
        # Ensure torch.cuda.is_available() returns False
        if hasattr(torch, 'cuda'):
            torch.cuda.is_available = lambda: False
            if hasattr(torch.cuda, '_is_available'):
                torch.cuda._is_available = lambda: False
        
        # Create or patch torch.xpu module with mock implementations
        if not hasattr(torch, 'xpu'):
            # Define a complete mock XPU module
            class XpuModule:
                def is_available(self):
                    return False
                def has_fp64_dtype(self):
                    return False
                def device_count(self):
                    return 0
                def current_device(self):
                    return 0
                def get_device_name(self, device=None):
                    return "CI_MOCK_XPU_DEVICE"
                def device(self, index=0):
                    return torch.device("cpu")
                    
            torch.xpu = XpuModule()
        else:
            # Patch existing torch.xpu module
            torch.xpu.is_available = lambda: False
            if not hasattr(torch.xpu, 'has_fp64_dtype'):
                torch.xpu.has_fp64_dtype = lambda: False
            if not hasattr(torch.xpu, 'device_count'):
                torch.xpu.device_count = lambda: 0
            if not hasattr(torch.xpu, 'current_device'):
                torch.xpu.current_device = lambda: 0
            if not hasattr(torch.xpu, 'get_device_name'):
                torch.xpu.get_device_name = lambda device=None: "CI_MOCK_XPU_DEVICE"
            if not hasattr(torch.xpu, 'device'):
                torch.xpu.device = lambda index=0: torch.device("cpu")
        
        print("Successfully patched torch for CPU-only mode")
    except ImportError:
        print("Failed to import torch")
        pass

# Apply patches when this module is imported
patch_torch()
EOF

# Make it executable and copy to site-packages
chmod +x sitecustomize.py
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
cp sitecustomize.py $SITE_PACKAGES/

# Create intel_extension_for_pytorch stub
mkdir -p $SITE_PACKAGES/intel_extension_for_pytorch
cat > $SITE_PACKAGES/intel_extension_for_pytorch/__init__.py << 'EOF'
# Dummy stub to satisfy intel_extension_for_pytorch import
# Provides minimal implementation for CI

class IpexGradScaler:
    def scale(self, loss): 
        return loss
    def step(self, optimizer): 
        pass
    def update(self): 
        pass

def optimize(model, dtype=None, inplace=False, **kwargs):
    return model

def quantization_aware_training():
    return None

# Critical function needed by xpu_hijacks.py
def has_xpu():
    return False

# Additional functions and attributes often used with IPEX
def enable_auto_mixed_precision(*args, **kwargs):
    return None

def enable_onednn_fusion(*args, **kwargs):
    return None
    
xpu = None  # This attribute may be accessed directly

def get_device_name(*args, **kwargs):
    return "CPU"
EOF

# Verify the modules
echo "Verifying CPU mode patches..."
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('torch.xpu available:', hasattr(torch, 'xpu'))"
python -c "import intel_extension_for_pytorch as ipex; print('ipex.has_xpu():', ipex.has_xpu())"

echo "CPU mode patches created successfully!" 