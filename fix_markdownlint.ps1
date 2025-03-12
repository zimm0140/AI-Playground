# PowerShell script to fix common markdown linting issues

# Function to fix ordered lists (MD029)
function Fix-OrderedLists {
    param (
        [string]$content
    )
    
    $lines = $content -split "`n"
    $newLines = @()
    $inList = $false
    $listCounter = 0
    
    foreach ($line in $lines) {
        # Check if this line starts a list item
        if ($line -match '^\s*1\.\s') {
            if (!$inList) {
                $inList = $true
                $listCounter = 1
                $newLines += $line
            }
            else {
                $listCounter++
                $newLine = $line -replace '^\s*1\.', "$listCounter."
                $newLines += $newLine
            }
        }
        else {
            # If the line doesn't match a list item pattern, check if we're exiting a list
            if ($line -match '^\s*$' -or $line -match '^#') {
                $inList = $false
                $listCounter = 0
            }
            $newLines += $line
        }
    }
    
    return $newLines -join "`n"
}

# Function to fix strong style (MD050)
function Fix-StrongStyle {
    param (
        [string]$content
    )
    
    # Replace __strong text__ with **strong text**
    $content = $content -replace '__(.+?)__', '**$1**'
    
    return $content
}

# Function to fix emphasis style (MD049)
function Fix-EmphasisStyle {
    param (
        [string]$content
    )
    
    # Replace *emphasized text* with _emphasized text_
    $content = $content -replace '\*([^\*]+)\*', '_$1_'
    
    return $content
}

# Function to add alt text to images (MD045)
function Fix-ImageAltText {
    param (
        [string]$content
    )
    
    # Replace ![](image.ext) with ![Image alt text](image.ext)
    $content = $content -replace '!\[\]\(([^)]+)\)', '![Image description]($1)'
    
    return $content
}

# Process each markdown file
$mdFiles = Get-ChildItem -Path . -Filter *.md -Recurse

foreach ($file in $mdFiles) {
    Write-Host "Processing $($file.FullName)"
    $content = Get-Content -Path $file.FullName -Raw
    
    # Apply fixes
    $content = Fix-OrderedLists -content $content
    $content = Fix-StrongStyle -content $content
    $content = Fix-EmphasisStyle -content $content
    $content = Fix-ImageAltText -content $content
    
    # Save back to file
    Set-Content -Path $file.FullName -Value $content
}

Write-Host "Finished processing all markdown files!"
