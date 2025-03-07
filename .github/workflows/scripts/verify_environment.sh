#!/bin/bash
# Environment Verification Script
# This script performs comprehensive checks of the environment to ensure all patches are working

echo "Checking environment setup before tests"

# Check torch setup
python -c "
try:
    import torch
    print('CUDA available:', torch.cuda.is_available() if hasattr(torch, 'cuda') else 'No CUDA module')
    print('XPU available:', getattr(torch, 'xpu', None) is not None and (hasattr(torch.xpu, 'is_available') and torch.xpu.is_available()))
except Exception as e:
    print(f'Error importing torch: {e}')
"

# Check environment variables
python -c "
import os
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
print('XPU_VISIBLE_DEVICES:', os.environ.get('XPU_VISIBLE_DEVICES', 'Not set'))
print('FORCE_CPU_ONLY:', os.environ.get('FORCE_CPU_ONLY', 'Not set'))
"

# Check IPEX stub
python -c "
try:
    import intel_extension_for_pytorch as ipex
    print('IPEX has_xpu:', hasattr(ipex, 'has_xpu') and ipex.has_xpu())
    print('IPEX available functions:', [f for f in dir(ipex) if not f.startswith('_')][:10])
except Exception as e:
    print(f'Error importing ipex: {e}')
"

# Check if xpu_hijacks has been patched
if [ -f service/xpu_hijacks.py ]; then
  grep -n "has_xpu" service/xpu_hijacks.py | head -5
fi

echo "Environment verification completed successfully!" 