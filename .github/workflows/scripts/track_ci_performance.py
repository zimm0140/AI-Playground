#!/usr/bin/env python3
"""
Track CI Performance Metrics

This script tracks CI performance metrics like job duration, step duration,
cache hit rates, and resource utilization. It can be run as part of the CI
workflow to collect and store metrics over time.

The metrics are stored in JSON files for each workflow run and can be used
to generate reports and visualizations to identify performance trends.
"""

import os
import json
import time
import argparse
import platform
import subprocess
import psutil
from datetime import datetime


def collect_metrics(workflow_name, job_name, output_dir):
    """
    Collect various performance metrics about the CI environment and run

    Args:
        workflow_name: Name of the current workflow
        job_name: Name of the current job
        output_dir: Directory to store metrics

    Returns:
        dict: Collected metrics
    """
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "workflow": workflow_name,
        "job": job_name,
        "system": {
            "platform": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "resources": {
            "cpu_count": os.cpu_count(),
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_used_percent": psutil.virtual_memory().percent,
            "disk_total": psutil.disk_usage("/").total,
            "disk_free": psutil.disk_usage("/").free,
            "disk_used_percent": psutil.disk_usage("/").percent,
        },
        "git": {
            "commit": get_git_commit(),
            "branch": get_git_branch(),
        },
        "duration": {
            "start_time": time.time(),  # Will be updated at the end
            "end_time": None,  # Will be updated at the end
            "total_seconds": None,  # Will be updated at the end
        },
        "cache": {
            "hits": 0,  # Will be populated from GitHub env if available
            "misses": 0,  # Will be populated from GitHub env if available
        },
        "steps": {},  # Will be populated during/after run
    }

    # Try to get cache metrics from environment variables if available
    cache_hits = os.environ.get("CACHE_HITS", "0")
    cache_misses = os.environ.get("CACHE_MISSES", "0")
    try:
        metrics["cache"]["hits"] = int(cache_hits)
        metrics["cache"]["misses"] = int(cache_misses)
    except ValueError:
        pass

    return metrics


def update_metrics(metrics, step_name=None, step_duration=None, status="running"):
    """
    Update metrics with new information

    Args:
        metrics: Existing metrics dict
        step_name: Name of the current step (if any)
        step_duration: Duration of the step in seconds (if completed)
        status: Status of the workflow/job/step

    Returns:
        dict: Updated metrics
    """
    # If a step is being recorded
    if step_name:
        if step_name not in metrics["steps"]:
            metrics["steps"][step_name] = {
                "start_time": time.time(),
                "end_time": None,
                "duration": None,
                "status": "running",
            }

        # If step duration is provided, the step has completed
        if step_duration is not None:
            metrics["steps"][step_name]["end_time"] = time.time()
            metrics["steps"][step_name]["duration"] = step_duration
            metrics["steps"][step_name]["status"] = status

    # Update resource metrics
    metrics["resources"]["cpu_usage_percent"] = psutil.cpu_percent(interval=0.1)
    metrics["resources"]["memory_used_percent"] = psutil.virtual_memory().percent
    metrics["resources"]["disk_used_percent"] = psutil.disk_usage("/").percent

    # If the overall job/workflow is marked as completed
    if status in ["success", "failure", "cancelled", "completed"]:
        metrics["duration"]["end_time"] = time.time()
        metrics["duration"]["total_seconds"] = (
            metrics["duration"]["end_time"] - metrics["duration"]["start_time"]
        )

    return metrics


def save_metrics(metrics, output_dir):
    """
    Save metrics to a JSON file

    Args:
        metrics: Metrics dict to save
        output_dir: Directory to save metrics to
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique filename with timestamp and workflow/job name
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    workflow_name = metrics["workflow"].replace(" ", "_").lower()
    job_name = metrics["job"].replace(" ", "_").lower()
    filename = f"ci_metrics_{timestamp}_{workflow_name}_{job_name}.json"

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {filepath}")

    # Also write a summary markdown file for GitHub
    summary_path = os.path.join(
        output_dir, f"ci_metrics_summary_{workflow_name}_{job_name}.md"
    )
    write_summary_markdown(metrics, summary_path)

    return filepath


def write_summary_markdown(metrics, output_path):
    """
    Write a markdown summary of the metrics

    Args:
        metrics: Metrics dict
        output_path: Path to write the summary to
    """
    with open(output_path, "w") as f:
        f.write("# CI Performance Metrics\n\n")
        f.write("## Job Information\n")
        f.write(f"- **Workflow**: {metrics['workflow']}\n")
        f.write(f"- **Job**: {metrics['job']}\n")
        f.write(f"- **Timestamp**: {metrics['timestamp']}\n")
        f.write(f"- **Git Branch**: {metrics['git']['branch']}\n")
        f.write(f"- **Git Commit**: {metrics['git']['commit']}\n\n")

        # Duration
        if metrics["duration"]["total_seconds"]:
            mins, secs = divmod(metrics["duration"]["total_seconds"], 60)
            f.write("## Duration\n")
            f.write(f"- **Total Time**: {int(mins)}m {int(secs)}s\n\n")

        # Cache stats
        f.write("## Cache Performance\n")
        total_cache_requests = metrics["cache"]["hits"] + metrics["cache"]["misses"]
        hit_rate = (
            metrics["cache"]["hits"] / total_cache_requests * 100
            if total_cache_requests > 0
            else 0
        )
        f.write(f"- **Cache Hits**: {metrics['cache']['hits']}\n")
        f.write(f"- **Cache Misses**: {metrics['cache']['misses']}\n")
        f.write(f"- **Hit Rate**: {hit_rate:.1f}%\n\n")

        # Resources
        f.write("## System Resources\n")
        f.write(f"- **CPU Usage**: {metrics['resources']['cpu_usage_percent']:.1f}%\n")
        memory_gb = metrics["resources"]["memory_total"] / (1024 * 1024 * 1024)
        memory_used = metrics["resources"]["memory_used_percent"]
        f.write(f"- **Memory**: {memory_used:.1f}% of {memory_gb:.1f} GB\n")
        disk_gb = metrics["resources"]["disk_total"] / (1024 * 1024 * 1024)
        disk_used = metrics["resources"]["disk_used_percent"]
        f.write(f"- **Disk**: {disk_used:.1f}% of {disk_gb:.1f} GB\n\n")

        # Step performance
        if metrics["steps"]:
            f.write("## Step Performance\n")
            f.write("| Step | Duration | Status |\n")
            f.write("|------|----------|--------|\n")

            for step_name, step_data in metrics["steps"].items():
                duration = step_data.get("duration")
                if duration:
                    mins, secs = divmod(duration, 60)
                    duration_str = f"{int(mins)}m {int(secs)}s"
                else:
                    duration_str = "In progress"

                status = step_data.get("status", "unknown")
                status_icon = (
                    "✅"
                    if status == "success"
                    else "❌"
                    if status == "failure"
                    else "⏳"
                )

                f.write(f"| {step_name} | {duration_str} | {status_icon} |\n")


def get_git_commit():
    """Get the current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def get_git_branch():
    """Get the current git branch name"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def initialize_from_github_env():
    """
    Initialize metrics from GitHub environment variables
    """
    workflow_name = os.environ.get("GITHUB_WORKFLOW", "unknown")
    job_name = os.environ.get("GITHUB_JOB", "unknown")

    # Default output directory for metrics
    output_dir = os.environ.get("METRICS_DIR", "ci_artifacts/metrics")

    return workflow_name, job_name, output_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Track CI performance metrics")

    parser.add_argument(
        "--workflow",
        help="Name of the workflow",
        default=os.environ.get("GITHUB_WORKFLOW", "unknown"),
    )

    parser.add_argument(
        "--job", help="Name of the job", default=os.environ.get("GITHUB_JOB", "unknown")
    )

    parser.add_argument(
        "--output-dir",
        help="Directory to store metrics",
        default=os.environ.get("METRICS_DIR", "ci_artifacts/metrics"),
    )

    parser.add_argument("--step", help="Name of the step being tracked", default=None)

    parser.add_argument(
        "--step-duration",
        help="Duration of the step in seconds",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--status",
        help="Status of the workflow/job/step",
        choices=["running", "success", "failure", "cancelled", "completed"],
        default="running",
    )

    parser.add_argument(
        "--init", help="Initialize metrics collection", action="store_true"
    )

    parser.add_argument(
        "--finalize", help="Finalize metrics collection", action="store_true"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    metrics_file = os.path.join(
        args.output_dir, f"current_metrics_{args.workflow}_{args.job}.json"
    )

    if args.init:
        # Initialize new metrics
        metrics = collect_metrics(args.workflow, args.job, args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Initialized metrics tracking for {args.workflow}/{args.job}")
        return

    # Load existing metrics if available
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    else:
        # Start new metrics if file doesn't exist
        metrics = collect_metrics(args.workflow, args.job, args.output_dir)

    # Update with current information
    metrics = update_metrics(
        metrics,
        step_name=args.step,
        step_duration=args.step_duration,
        status=args.status,
    )

    # Save updated metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # If finalizing, save final metrics and cleanup current file
    if args.finalize:
        final_path = save_metrics(metrics, args.output_dir)
        print(f"Finalized metrics: {final_path}")

        # Try to remove the temporary file
        try:
            os.remove(metrics_file)
        except OSError:
            pass


if __name__ == "__main__":
    main()
