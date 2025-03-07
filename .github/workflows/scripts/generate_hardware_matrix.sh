#!/bin/bash
# Hardware Compatibility Matrix Generator
# This script analyzes requirements files and generates a comprehensive hardware matrix

echo "Generating hardware compatibility matrix..."
mkdir -p ci_artifacts/hardware_matrix

# Define hardware platforms and their requirements files
declare -A HW_PLATFORMS
HW_PLATFORMS=(
  ["Standard CPU"]="service/requirements.txt"
  ["Intel Arc (ACM)"]="service/requirements-acm.txt" 
  ["Intel Lunar Lake (LNL)"]="service/requirements-lnl.txt"
  ["Intel Battlemage (BMG)"]="service/requirements-bmg.txt"
  ["Intel Meteor Lake (MTL)"]="service/requirements-mtl.txt"
  ["Intel Arc Alchemist (ARL_H)"]="service/requirements-arl_h.txt"
  ["Intel Level Zero"]="service/requirements-ls_level_zero.txt"
  ["OpenVINO"]="OpenVINO/requirements.txt"
  ["LlamaCPP"]="LlamaCPP/requirements.txt"
)

# Extract feature support from requirements files
extract_features() {
  local req_file="$1"
  local features=()
  
  if [ ! -f "$req_file" ]; then
    echo "Not Found"
    return
  fi
  
  # Detect key dependencies
  if grep -q "torch" "$req_file"; then features+=("PyTorch"); fi
  if grep -q "intel-extension-for-pytorch" "$req_file"; then features+=("IPEX"); fi
  if grep -q "ipex_llm" "$req_file"; then features+=("IPEX-LLM"); fi
  if grep -q "bigdl" "$req_file"; then features+=("BigDL"); fi
  if grep -q "openvino" "$req_file"; then features+=("OpenVINO"); fi
  if grep -q "diffusers" "$req_file"; then features+=("Diffusers"); fi
  if grep -q "transformers" "$req_file"; then features+=("Transformers"); fi
  if grep -q "langchain" "$req_file"; then features+=("LangChain"); fi
  
  # Extract version information
  local torch_ver=$(grep "torch==" "$req_file" | grep -o "=[0-9][0-9.]*" | head -1 | tr -d '=')
  local ipex_ver=$(grep "intel-extension-for-pytorch" "$req_file" | grep -o "=[0-9][0-9.]*" | head -1 | tr -d '=')
  
  if [ -n "$torch_ver" ]; then features+=("PyTorch $torch_ver"); fi
  if [ -n "$ipex_ver" ]; then features+=("IPEX $ipex_ver"); fi
  
  # Join features with commas
  local feature_string=$(IFS=", "; echo "${features[*]}")
  
  if [ -z "$feature_string" ]; then
    echo "Basic support"
  else
    echo "$feature_string"
  fi
}

# Detect acceleration support
detect_acceleration() {
  local req_file="$1"
  local accel=()
  
  if [ ! -f "$req_file" ]; then
    echo "None"
    return
  fi
  
  if grep -q "xpu" "$req_file"; then accel+=("Intel XPU"); fi
  if grep -q "cuda" "$req_file"; then accel+=("NVIDIA CUDA"); fi
  if grep -q "rocm" "$req_file"; then accel+=("AMD ROCm"); fi
  if grep -q "dpcpp" "$req_file"; then accel+=("Intel DPCPP"); fi
  if grep -q "level-zero" "$req_file" || grep -q "level_zero" "$req_file"; then accel+=("Level Zero"); fi
  
  # Join acceleration with commas
  local accel_string=$(IFS=", "; echo "${accel[*]}")
  
  if [ -z "$accel_string" ]; then
    echo "CPU only"
  else
    echo "$accel_string"
  fi
}

# Determine OS compatibility
detect_os_compatibility() {
  local req_file="$1"
  
  if [ ! -f "$req_file" ]; then
    echo "Unknown"
    return
  fi
  
  local os_support=("Linux")
  
  # Check for Windows-specific packages or comments
  if grep -i -E "win|windows" "$req_file"; then
    os_support+=("Windows")
  fi
  
  # Check for macOS-specific packages or comments
  if grep -i -E "mac|darwin|osx" "$req_file"; then
    os_support+=("macOS")
  fi
  
  # Join OS with commas
  local os_string=$(IFS=", "; echo "${os_support[*]}")
  echo "$os_string"
}

# Create the header for the markdown table
{
  echo "# Hardware Compatibility Matrix"
  echo ""
  echo "This matrix provides an overview of hardware platforms supported by the AI Playground service,"
  echo "including details about required dependencies, acceleration types, and OS compatibility."
  echo ""
  echo "## Platform Support"
  echo ""
  echo "| Platform | Features | Acceleration | OS Support | Status |"
  echo "|----------|----------|--------------|------------|--------|"
  
  # Add rows for each hardware platform
  for platform in "${!HW_PLATFORMS[@]}"; do
    req_file="${HW_PLATFORMS[$platform]}"
    
    features=$(extract_features "$req_file")
    acceleration=$(detect_acceleration "$req_file")
    os_support=$(detect_os_compatibility "$req_file")
    
    # Determine status based on existence of requirements file
    if [ -f "$req_file" ]; then
      status="✅ Supported"
    else
      status="❌ Not supported"
    fi
    
    echo "| $platform | $features | $acceleration | $os_support | $status |"
  done
  
  echo ""
  echo "## Integration Details"
  echo ""
  echo "### Feature Explanation"
  echo ""
  echo "- **PyTorch**: Base deep learning framework"
  echo "- **IPEX**: Intel PyTorch Extensions for optimized performance on Intel hardware"
  echo "- **IPEX-LLM**: Extensions specifically for large language models on Intel hardware"
  echo "- **BigDL**: Distributed deep learning library for Intel hardware"
  echo "- **OpenVINO**: Intel's toolkit for optimized inference"
  echo "- **Diffusers**: Library for state-of-the-art diffusion models"
  echo "- **Transformers**: Hugging Face transformers library for NLP models"
  echo "- **LangChain**: Framework for developing applications powered by language models"
  echo ""
  echo "### Acceleration Types"
  echo ""
  echo "- **Intel XPU**: Intel's unified API for accelerated computing across CPUs and GPUs"
  echo "- **Intel DPCPP**: Data Parallel C++ runtime for Intel architectures"
  echo "- **Level Zero**: Low-level direct-to-metal interface for Intel GPUs"
  echo "- **NVIDIA CUDA**: Parallel computing platform for NVIDIA GPUs"
  echo "- **AMD ROCm**: Open software platform for GPU computing on AMD hardware"
  echo ""
  echo "Generated on: $(date)"
  echo ""
  echo "*This document is automatically generated by CI and reflects the current state of hardware compatibility.*"
} > ci_artifacts/hardware_matrix/compatibility_matrix.md

echo "Hardware compatibility matrix generated at ci_artifacts/hardware_matrix/compatibility_matrix.md"

# Add to GitHub step summary
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
  {
    echo "## Hardware Compatibility Matrix"
    echo ""
    echo "A comprehensive hardware compatibility matrix has been generated:"
    echo ""
    echo "- Total platforms analyzed: ${#HW_PLATFORMS[@]}"
    echo "- Matrix available in artifacts: ci_artifacts/hardware_matrix/compatibility_matrix.md"
    echo ""
    echo "### Quick Platform Summary"
    echo ""
    
    # Count supported platforms
    supported=0
    for platform in "${!HW_PLATFORMS[@]}"; do
      if [ -f "${HW_PLATFORMS[$platform]}" ]; then
        ((supported++))
      fi
    done
    
    echo "- Supported platforms: $supported of ${#HW_PLATFORMS[@]}"
  } >> $GITHUB_STEP_SUMMARY
fi 