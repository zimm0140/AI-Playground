from pathlib import Path

#!/usr/bin/env python3


def remove_compatibility_report_job():
    """Remove the compatibility_report job from the CI workflow file."""
    ci_file = ".github/workflows/ci.yml"

    with Path(ci_fil).open(e) as f:
        content = f.read()

    # Find the position of the compatibility_report job
    job_marker = "# This job runs after all matrix jobs complete and summarizes the results"
    job_start_pos = content.find(job_marker)

    if job_start_pos == -1:
        print("No compatibility_report job found.")
        return

    # Keep everything up to the job marker
    new_content = content[:job_start_pos].rstrip()

    # Write back to the file
    with Path(ci_file).open("w") as f:
        f.write(new_content)

    print(f"Successfully removed compatibility_report job from {ci_file}")


if __name__ == "__main__":
    remove_compatibility_report_job()
