#!/bin/bash
# API Documentation Generator
# This script generates API documentation from Python docstrings

echo "Generating API documentation..."

# Install documentation generation tools
python -m pip install pdoc3

# Create the docs directory
mkdir -p api_docs

# Generate basic API docs from docstrings
python -c "
import os
import sys

# Set up paths so service module can be imported
sys.path.insert(0, os.getcwd())

try:
    # Use pdoc to generate documentation
    import pdoc
    
    # Configure pdoc options
    pdoc.tpl_dir = 'api_docs'
    
    # Generate HTML documentation
    modules_to_document = []
    if os.path.exists('service'):
        for file in os.listdir('service'):
            if file.endswith('.py') and not file.startswith('__'):
                module_name = f'service.{file[:-3]}'
                modules_to_document.append(module_name)
    
    # Print modules being documented
    print(f'Documenting modules: {modules_to_document}')
    
    for module in modules_to_document:
        try:
            # Create HTML documentation
            html = pdoc.html(module)
            
            # Save to file
            module_file = module.replace('.', '/') + '.html'
            os.makedirs(os.path.dirname(f'api_docs/{module_file}'), exist_ok=True)
            with open(f'api_docs/{module_file}', 'w') as f:
                f.write(html)
            
            print(f'Successfully documented {module}')
        except Exception as e:
            print(f'Error documenting {module}: {e}')
except Exception as e:
    print(f'Failed to generate documentation: {e}')
    # Create a placeholder file
    with open('api_docs/index.html', 'w') as f:
        f.write('<html><body><h1>API Documentation</h1><p>Placeholder for API documentation. Could not generate full docs in CI environment.</p></body></html>')
"

# Create an index file
echo '<html><head><title>API Documentation</title></head><body><h1>API Documentation</h1><ul>' > api_docs/index.html

# List all documented modules
for file in $(find api_docs -name "*.html" -not -path "*/\.*" | sort); do
  module=$(basename $file .html)
  if [ "$module" != "index" ]; then
    echo "<li><a href=\"$file\">$module</a></li>" >> api_docs/index.html
  fi
done

echo '</ul></body></html>' >> api_docs/index.html

echo "API documentation generated in api_docs directory" 