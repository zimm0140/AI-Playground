const fs = require('fs');
const path = require('path');
const os = require('os');

/**
 * This script ensures that the npm global directory exists.
 * It prevents ENOENT errors when running npm/npx commands in CI environments.
 */
function ensureNpmDirectory() {
  try {
    // Get the npm directory path
    const npmDir = path.join(os.homedir(), 'AppData', 'Roaming', 'npm');
    
    // Check if the directory exists
    if (!fs.existsSync(npmDir)) {
      console.log(`Creating npm directory: ${npmDir}`);
      fs.mkdirSync(npmDir, { recursive: true });
      console.log('npm directory created successfully.');
    } else {
      console.log(`npm directory already exists: ${npmDir}`);
    }
  } catch (error) {
    console.error('Error ensuring npm directory exists:', error);
    process.exit(1);
  }
}

// Execute the function
ensureNpmDirectory(); 