#!/usr/bin/env python3
"""
Generate a summary of workflow validation and requirements analysis results
for GitHub Actions workflow step summary.
"""

import os
import json

def main():
    """Generate summary from validation and requirements results"""

    # Get the path to the GitHub step summary file
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY', 'summary.md')

    # Add validation results section
    with open(summary_file, 'a') as summary:
        summary.write("## Validation Results\n\n")

    # Check if validation results exist
    if os.path.exists('ci_artifacts/workflow_validation/workflow_validation_results.json'):
        with open('ci_artifacts/workflow_validation/workflow_validation_results.json', 'r') as f:
            results = json.load(f)
            
        with open(summary_file, 'a') as summary:
            summary.write(f"Total workflows: {results['summary']['total_workflows']}\n\n")
            summary.write(f"Valid workflows: {results['summary']['valid_workflows']}\n\n")
            summary.write(f"Invalid workflows: {results['summary']['invalid_workflows']}\n\n")
            
            if results['summary']['invalid_workflows'] > 0:
                summary.write('### Invalid Workflows\n\n')
                for workflow in results['workflows']:
                    if not workflow['is_valid']:
                        issue_count = len(workflow['issues'])
                        summary.write(f"- **{workflow['filename']}**: {issue_count} issues\n\n")
    else:
        with open(summary_file, 'a') as summary:
            summary.write("No validation results found\n\n")
    
    # Add requirements analysis section
    with open(summary_file, 'a') as summary:
        summary.write("## Requirements Analysis\n\n")

    # Check if requirements results exist
    if os.path.exists('ci_artifacts/workflow_requirements/workflow_requirements_results.json'):
        with open('ci_artifacts/workflow_requirements/workflow_requirements_results.json', 'r') as f:
            results = json.load(f)
            
        with open(summary_file, 'a') as summary:
            summary.write(f"Analyzed workflows: {results['summary']['analyzed_workflows']} of {results['summary']['total_workflows']}\n\n")
            
            if 'models' in results['aggregate'] and results['aggregate']['models']:
                summary.write('### Top Models\n\n')
                sorted_models = sorted(results['aggregate']['models'].items(), key=lambda x: x[1], reverse=True)[:5]
                for model, count in sorted_models:
                    summary.write(f"- {model}: Used in {count} workflows\n\n")
            
            if 'custom_nodes' in results['aggregate'] and results['aggregate']['custom_nodes']:
                summary.write('### Top Custom Node Extensions\n\n')
                sorted_nodes = sorted(results['aggregate']['custom_nodes'].items(), key=lambda x: x[1], reverse=True)[:5]
                for node, count in sorted_nodes:
                    summary.write(f"- {node}: Used in {count} workflows\n\n")
    else:
        with open(summary_file, 'a') as summary:
            summary.write("No requirements analysis results found\n\n")

if __name__ == "__main__":
    main() 