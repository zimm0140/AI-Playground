#!/usr/bin/env python
"""
API Documentation Summary Generator

This script analyzes the web_api.py file and generates a concise API
documentation summary in Markdown format, organized by endpoint function.
"""

import os
import re
import sys
from collections import defaultdict
import inspect
import argparse

def extract_endpoints(file_path):
    """Extract API endpoints from a Flask/apiflask web API file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patterns to match route definitions
    route_patterns = [
        r'@app\.(?:route|get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"](?:,\s*methods=\[(.*?)\])?\)',
        r'@app\.(?:input|output)\s*\((.*?)\)'
    ]
    
    # Pattern to match function definitions
    func_pattern = r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):'
    
    # Pattern to match docstrings
    docstring_pattern = r'"""(.*?)"""'
    
    # Extract all route definitions and their associated functions
    endpoints = []
    
    # Split file into function blocks
    func_blocks = re.split(r'\ndef\s+', content)
    
    for i, block in enumerate(func_blocks[1:], 1):  # Skip first block (imports, etc.)
        # Get function name and arguments
        func_match = re.match(r'([a-zA-Z0-9_]+)\s*\((.*?)\):', block, re.DOTALL)
        if not func_match:
            continue
        
        func_name = func_match.group(1)
        func_args = func_match.group(2).strip()
        
        # Look for route definitions above this function in the previous block
        prev_block = func_blocks[i-1] if i > 0 else ""
        route_defs = []
        
        for pattern in route_patterns:
            for match in re.finditer(pattern, prev_block + block[:func_match.start()], re.DOTALL):
                route_defs.append(match.group(0))
        
        # Extract routes
        routes = []
        for route_def in route_defs:
            route_match = re.search(r'@app\.(?:route|get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]', route_def)
            if route_match:
                route = route_match.group(1)
                
                # Determine HTTP method
                method_match = re.search(r'methods=\[(.*?)\]', route_def)
                methods = []
                if method_match:
                    methods = [m.strip(' "\'') for m in method_match.group(1).split(',')]
                else:
                    # Infer from decorator
                    if '@app.get' in route_def:
                        methods = ['GET']
                    elif '@app.post' in route_def:
                        methods = ['POST']
                    elif '@app.put' in route_def:
                        methods = ['PUT']
                    elif '@app.delete' in route_def:
                        methods = ['DELETE']
                    elif '@app.patch' in route_def:
                        methods = ['PATCH']
                    else:
                        methods = ['GET']  # Default to GET
                
                routes.append((route, methods))
        
        # Extract docstring if present
        docstring = ""
        docstring_match = re.search(docstring_pattern, block, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
        
        # Input/output annotations
        input_schema = None
        output_schema = None
        
        for route_def in route_defs:
            if '@app.input' in route_def:
                input_match = re.search(r'@app\.input\s*\(\s*([^,)]+)', route_def)
                if input_match:
                    input_schema = input_match.group(1).strip()
            
            if '@app.output' in route_def:
                output_match = re.search(r'@app\.output\s*\(\s*([^,)]+)', route_def)
                if output_match:
                    output_schema = output_match.group(1).strip()
        
        # Add to endpoints list if we found routes
        if routes:
            endpoints.append({
                'function': func_name,
                'routes': routes,
                'args': func_args,
                'docstring': docstring,
                'input_schema': input_schema,
                'output_schema': output_schema
            })
    
    return endpoints

def generate_markdown(endpoints, output_file=None):
    """Generate markdown documentation from the extracted endpoints."""
    if not endpoints:
        print("No endpoints found.")
        return
    
    # Organize endpoints by category based on URL prefix
    categories = defaultdict(list)
    
    for endpoint in endpoints:
        # Get first route for categorization
        if not endpoint['routes']:
            continue
            
        route = endpoint['routes'][0][0]
        parts = route.split('/')
        
        if len(parts) >= 3 and parts[1] == 'api':
            category = parts[2]
        else:
            category = 'other'
        
        categories[category].append(endpoint)
    
    # Generate markdown
    lines = [
        "# API Endpoints Summary",
        "",
        "This document provides a summary of all API endpoints in the application.",
        "",
        "## Table of Contents",
        ""
    ]
    
    # Add categories to TOC
    for category in sorted(categories.keys()):
        lines.append(f"- [{category.capitalize()} Endpoints](#{category}-endpoints)")
    
    lines.append("")
    
    # Add endpoints by category
    for category in sorted(categories.keys()):
        lines.append(f"## {category.capitalize()} Endpoints")
        lines.append("")
        
        for endpoint in categories[category]:
            function_name = endpoint['function']
            routes_str = []
            
            for route, methods in endpoint['routes']:
                methods_str = ", ".join(methods)
                routes_str.append(f"`{methods_str}` {route}")
            
            routes_formatted = "<br>".join(routes_str)
            
            lines.append(f"### {function_name}")
            lines.append("")
            lines.append(f"**Routes:** {routes_formatted}")
            lines.append("")
            
            if endpoint['input_schema']:
                lines.append(f"**Input Schema:** `{endpoint['input_schema']}`")
                lines.append("")
            
            if endpoint['output_schema']:
                lines.append(f"**Output Schema:** `{endpoint['output_schema']}`")
                lines.append("")
            
            if endpoint['docstring']:
                # Format the docstring, removing excessive indentation
                formatted_docstring = "\n".join(
                    line.strip() for line in endpoint['docstring'].split("\n")
                )
                lines.append("**Description:**")
                lines.append("")
                lines.append(formatted_docstring)
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    # Add metadata
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This documentation was automatically generated by the CI process.*")
    lines.append(f"*Generated on: {os.popen('date').read().strip()}*")
    
    # Write to file or stdout
    markdown = "\n".join(lines)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        print(f"API documentation written to {output_file}")
    else:
        print(markdown)
    
    return markdown

def generate_github_summary(endpoints):
    """Generate a GitHub step summary from the extracted endpoints."""
    if not os.environ.get('GITHUB_STEP_SUMMARY'):
        return
    
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
        f.write("## API Endpoints Summary\n\n")
        
        # Count endpoints by category and method
        categories = defaultdict(int)
        methods = defaultdict(int)
        schemas = 0
        
        for endpoint in endpoints:
            for route, methods_list in endpoint['routes']:
                parts = route.split('/')
                
                if len(parts) >= 3 and parts[1] == 'api':
                    category = parts[2]
                else:
                    category = 'other'
                
                categories[category] += 1
                
                for method in methods_list:
                    methods[method] += 1
            
            if endpoint['input_schema'] or endpoint['output_schema']:
                schemas += 1
        
        # Write endpoint counts
        f.write(f"Total API endpoints: **{len(endpoints)}**\n\n")
        
        # Write category table
        f.write("### Endpoints by Category\n\n")
        f.write("| Category | Count |\n")
        f.write("|----------|-------|\n")
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {category.capitalize()} | {count} |\n")
        
        f.write("\n")
        
        # Write methods table
        f.write("### Endpoints by HTTP Method\n\n")
        f.write("| Method | Count |\n")
        f.write("|--------|-------|\n")
        
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {method} | {count} |\n")
        
        f.write("\n")
        f.write(f"Endpoints with schema definitions: **{schemas}**\n\n")
        f.write("See API documentation artifact for complete details.\n")

def main():
    parser = argparse.ArgumentParser(description="Generate API documentation summary")
    parser.add_argument("--input", default="service/web_api.py", help="Path to the web API file")
    parser.add_argument("--output", default="ci_artifacts/api_summary/api_endpoints.md", help="Output markdown file path")
    args = parser.parse_args()
    
    print(f"Generating API documentation from {args.input}...")
    endpoints = extract_endpoints(args.input)
    
    if endpoints:
        print(f"Found {len(endpoints)} API endpoints")
        generate_markdown(endpoints, args.output)
        generate_github_summary(endpoints)
    else:
        print("No API endpoints found")
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
                f.write("## API Endpoints\n\n")
                f.write("No API endpoints found in the codebase.\n")

if __name__ == "__main__":
    main() 