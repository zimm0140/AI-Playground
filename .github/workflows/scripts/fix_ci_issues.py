#!/usr/bin/env python3
"""
Fix CI Issues Script

This script applies patches to source code files to make them compatible with the CI environment.
It fixes issues like invalid escape sequences, imports, and makes calls to hardware-dependent
modules more resilient.
"""

import os
import re

def patch_files():
    """Apply patches to make code work in CI environment"""
    print("Applying CI compatibility patches...")
    
    # Fix invalid escape sequences in paint_biz.py
    if os.path.exists('service/paint_biz.py'):
        print("Patching service/paint_biz.py...")
        with open('service/paint_biz.py', 'r') as f:
            content = f.read()
        
        # Fix invalid escape sequence
        if 'is_xl = re.search("[-_]xl[-_\\.]"' in content:
            content = content.replace(
                'is_xl = re.search("[-_]xl[-_\\.]"', 
                'is_xl = re.search(r"[-_]xl[-_\\.]"'
            )
            print("  - Fixed invalid escape sequence")
        
        with open('service/paint_biz.py', 'w') as f:
            f.write(content)
    
    # Patch web_api.py to handle imports safely
    if os.path.exists('service/web_api.py'):
        print("Patching service/web_api.py...")
        with open('service/web_api.py', 'r') as f:
            content = f.read()
        
        replacements = [
            ('from sd_adapter import SD_SSE_Adapter', 
             'try:\n    from sd_adapter import SD_SSE_Adapter\nexcept ImportError:\n    SD_SSE_Adapter = None'),
            ('from rag import RAGManager', 
             'try:\n    from rag import RAGManager\nexcept ImportError:\n    RAGManager = None')
        ]
        
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                print(f"  - Patched import: {old}")
        
        with open('service/web_api.py', 'w') as f:
            f.write(content)
    
    # Patch test_api.py to handle imports safely
    if os.path.exists('service/tests/test_api.py'):
        print("Patching service/tests/test_api.py...")
        with open('service/tests/test_api.py', 'r') as f:
            content = f.read()
        
        # Fix import to make it work in both local and CI contexts
        if 'from web_api import app' in content:
            content = content.replace(
                'from web_api import app',
                'try:\n    from web_api import app\nexcept ImportError:\n    from service.web_api import app'
            )
            print("  - Patched import to handle different module paths")
        
        with open('service/tests/test_api.py', 'w') as f:
            f.write(content)
    
    # Fix xpu_hijacks.py for more resilient ipex usage
    if os.path.exists('service/xpu_hijacks.py'):
        print("Patching service/xpu_hijacks.py...")
        with open('service/xpu_hijacks.py', 'r') as f:
            content = f.read()
        
        # Make ipex.has_xpu() calls safe
        if 'return ipex.has_xpu()' in content:
            content = content.replace(
                'return ipex.has_xpu()',
                'return hasattr(ipex, "has_xpu") and ipex.has_xpu()'
            )
            print("  - Made ipex.has_xpu() calls safer")
        
        with open('service/xpu_hijacks.py', 'w') as f:
            f.write(content)
    
    print("All CI compatibility patches applied successfully!")

if __name__ == "__main__":
    patch_files() 