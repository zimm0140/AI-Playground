#!/usr/bin/env python
"""
Hardware Benchmark for AI Workloads

This script benchmarks common AI operations across different hardware configurations
to help determine the optimal setup for your specific workloads.

Usage:
    python hardware_benchmark.py [--output results.json] [--iterations 5]

Tests:
    - Matrix multiplication (PyTorch)
    - Convolution operations
    - Model inference (LangChain)
    - Image generation (Stable Diffusion)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch

from hardware_detection import detect_hardware_type


class HardwareBenchmark:
    """Benchmark AI operations across different hardware configurations"""

    def __init__(self, iterations: int = 5, warmup: int = 2):
        """
        Initialize the benchmark

        Args:
            iterations: Number of iterations for each benchmark
            warmup: Number of warmup iterations (not counted in results)
        """
        self.iterations = iterations
        self.warmup = warmup
        self.hardware_type = detect_hardware_type()
        self.results = {
            "hardware_type": self.hardware_type,
            "system_info": self._get_system_info(),
            "benchmarks": {},
        }

        # Configure hardware
        self.device = self._configure_hardware()

    def _get_system_info(self) -> dict[str, str]:
        """Get system information"""
        import platform

        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        }

        # Add GPU info if available
        if self.hardware_type == "acm":
            try:
                import intel_extension_for_pytorch as ipex

                info["ipex_version"] = ipex.__version__
            except ImportError:
                pass

        return info

    def _configure_hardware(self) -> torch.device:
        """Configure hardware and return appropriate device"""
        print(f"Detected hardware type: {self.hardware_type}")

        if self.hardware_type == "acm":
            try:
                import intel_extension_for_pytorch as ipex
                from service.xpu_hijacks import ipex_hijacks

                # Apply XPU hijacks
                ipex_hijacks()

                # Set XPU environment variables
                os.environ["SYCL_CACHE_PERSISTENT"] = "1"

                device = torch.device("xpu")
                print(f"Using Intel Arc GPU device: {device}")
                return device
            except ImportError:
                print("Intel XPU extensions not available, falling back to CPU")

        # Default to CPU
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
        return device

    def _time_function(self, func, *args, **kwargs) -> list[float]:
        """
        Time a function execution over multiple iterations

        Args:
            func: Function to time
            *args, **kwargs: Arguments to pass to the function

        Returns:
            List of execution times in seconds
        """
        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)

        # Sync if using GPU
        if self.device.type in ["cuda", "xpu"]:
            torch.cuda.synchronize() if self.device.type == "cuda" else torch.xpu.synchronize()

        # Actual benchmark
        times = []
        for _ in range(self.iterations):
            start_time = time.time()
            func(*args, **kwargs)

            # Sync if using GPU
            if self.device.type in ["cuda", "xpu"]:
                torch.cuda.synchronize() if self.device.type == "cuda" else torch.xpu.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

        return times

    def benchmark_matrix_multiplication(self, sizes: list[int] = None) -> dict[str, Any]:
        """
        Benchmark matrix multiplication

        Args:
            sizes: List of matrix sizes to benchmark

        Returns:
            Dictionary with benchmark results
        """
        if sizes is None:
            sizes = [1024, 2048, 4096]

        results = {}
        for size in sizes:
            print(f"Benchmarking {size}x{size} matrix multiplication...")

            # Create random matrices
            matrix_a = torch.rand(size, size, device=self.device)
            matrix_b = torch.rand(size, size, device=self.device)

            # Time matrix multiplication
            times = self._time_function(lambda a, b: torch.matmul(a, b), matrix_a, matrix_b)

            # Record results
            results[f"{size}x{size}"] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "times": times,
            }

        self.results["benchmarks"]["matrix_multiplication"] = results
        return results

    def benchmark_convolution(self, batch_sizes: list[int] = None) -> dict[str, Any]:
        """
        Benchmark convolution operations (common in CNNs)

        Args:
            batch_sizes: List of batch sizes to benchmark

        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32]

        results = {}

        # Create a simple convolutional layer
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(self.device)

        for batch_size in batch_sizes:
            print(f"Benchmarking convolution with batch size {batch_size}...")

            # Create random input
            input_tensor = torch.rand(batch_size, 3, 224, 224, device=self.device)

            # Time convolution
            times = self._time_function(lambda x: conv(x), input_tensor)

            # Record results
            results[f"batch_{batch_size}"] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "times": times,
            }

        self.results["benchmarks"]["convolution"] = results
        return results

    def benchmark_model_inference(
        self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    ) -> dict[str, Any]:
        """
        Benchmark model inference

        Args:
            model_name: Name of the model to benchmark

        Returns:
            Dictionary with benchmark results
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Benchmarking inference with {model_name}...")

            # Load model and tokenizer
            print("Loading model and tokenizer...")

            # Different loading based on hardware
            if self.hardware_type == "acm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, trust_remote_code=True
                ).to(self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=True, trust_remote_code=True
                ).to(self.device)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Prepare input
            input_text = "Explain quantum computing in simple terms."
            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)

            # Time inference
            def inference():
                with torch.no_grad():
                    outputs = model.generate(inputs.input_ids, max_new_tokens=128, do_sample=False)
                return outputs

            print("Running inference benchmark...")
            times = self._time_function(inference)

            # Record results
            results = {
                "model": model_name,
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "times": times,
            }

            self.results["benchmarks"]["model_inference"] = results
            return results

        except ImportError:
            print("Transformers library not available, skipping model inference benchmark")
            return {"error": "Transformers library not available"}

    def benchmark_stable_diffusion(
        self, prompt: str = "a photo of an astronaut riding a horse on mars"
    ) -> dict[str, Any]:
        """
        Benchmark Stable Diffusion image generation

        Args:
            prompt: Prompt for image generation

        Returns:
            Dictionary with benchmark results
        """
        try:
            from diffusers import StableDiffusionPipeline

            print("Benchmarking Stable Diffusion image generation...")

            # Load pipeline
            print("Loading Stable Diffusion pipeline...")

            # Different loading based on hardware
            if self.hardware_type == "acm":
                try:
                    import intel_extension_for_pytorch as ipex

                    pipeline = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5"
                    ).to(self.device)

                    # Optional: Optimize with IPEX
                    pipeline = ipex.optimize(pipeline)
                except ImportError:
                    print("Intel XPU extensions not available, loading standard pipeline")
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5"
                    ).to(self.device)
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
                ).to(self.device)

            # Time image generation
            def generate_image():
                with torch.no_grad():
                    image = pipeline(prompt, num_inference_steps=20).images[0]
                return image

            print("Running Stable Diffusion benchmark...")
            times = self._time_function(generate_image)

            # Record results
            results = {
                "prompt": prompt,
                "num_steps": 20,
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "times": times,
            }

            self.results["benchmarks"]["stable_diffusion"] = results
            return results

        except ImportError:
            print("Diffusers library not available, skipping Stable Diffusion benchmark")
            return {"error": "Diffusers library not available"}

    def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all benchmarks"""
        print(f"Running all benchmarks on {self.hardware_type} hardware...")

        # Run benchmarks
        self.benchmark_matrix_multiplication()
        self.benchmark_convolution()

        # Optional benchmarks (may not have dependencies)
        try:
            self.benchmark_model_inference()
        except Exception as e:
            print(f"Error in model inference benchmark: {e}")

        try:
            self.benchmark_stable_diffusion()
        except Exception as e:
            print(f"Error in Stable Diffusion benchmark: {e}")

        # Add summary
        self._add_summary()

        return self.results

    def _add_summary(self) -> None:
        """Add summary to results"""
        summary = {}

        for bench_name, bench_results in self.results["benchmarks"].items():
            if isinstance(bench_results, dict) and not bench_results.get("error"):
                if "mean" in bench_results:
                    # Direct result
                    summary[bench_name] = bench_results["mean"]
                else:
                    # Nested results, take the largest size
                    largest_size = list(bench_results.keys())[-1]
                    summary[bench_name] = bench_results[largest_size]["mean"]

        self.results["summary"] = summary

    def save_results(self, output_file: str) -> None:
        """Save results to file"""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {output_file}")

    def print_summary(self) -> None:
        """Print results summary"""
        print("\n===== BENCHMARK SUMMARY =====")
        print(f"Hardware type: {self.hardware_type}")

        if "summary" in self.results:
            for bench_name, mean_time in self.results["summary"].items():
                print(f"{bench_name}: {mean_time:.4f} seconds")
        else:
            print("No summary available. Run benchmarks first.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark AI operations across hardware")
    parser.add_argument(
        "--output", default="benchmark_results.json", help="Output file for results"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations for each benchmark"
    )
    parser.add_argument(
        "--matrix", action="store_true", help="Run only matrix multiplication benchmark"
    )
    parser.add_argument("--conv", action="store_true", help="Run only convolution benchmark")
    parser.add_argument("--model", action="store_true", help="Run only model inference benchmark")
    parser.add_argument("--sd", action="store_true", help="Run only Stable Diffusion benchmark")

    args = parser.parse_args()

    benchmark = HardwareBenchmark(iterations=args.iterations)

    # Determine which benchmarks to run
    run_specific = args.matrix or args.conv or args.model or args.sd

    if not run_specific:
        # Run all benchmarks if no specific one is requested
        benchmark.run_all_benchmarks()
    else:
        if args.matrix:
            benchmark.benchmark_matrix_multiplication()
        if args.conv:
            benchmark.benchmark_convolution()
        if args.model:
            benchmark.benchmark_model_inference()
        if args.sd:
            benchmark.benchmark_stable_diffusion()

    # Save and print results
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
