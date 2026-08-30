$ErrorActionPreference = "Stop"

# This script downloads and unpacks the PulsePPG model weights on Windows.

$URL = "https://zenodo.org/records/17345536/files/pulseppg_model_weights.zip?download=1"
$FILENAME = "pulseppg_model_weights.zip"

Write-Host "Downloading PulsePPG model weights..."
try {
    Invoke-WebRequest -Uri $URL -OutFile $FILENAME
}
catch {
    Write-Error "Error: Download failed. Please check the URL and your internet connection."
    exit 1
}

Write-Host "Download complete."

Write-Host "Unpacking the zip file..."
try {
    Expand-Archive -Path $FILENAME -DestinationPath "." -Force
}
catch {
    Write-Error "Error: Unpacking failed. The file might be corrupted or the archive extraction is unavailable."
    exit 1
}

Write-Host "Unpacking complete."
Write-Host "Cleaning up the downloaded zip file..."
Remove-Item -Path $FILENAME -Force

Write-Host "Done! The model weights have been downloaded and unpacked."
