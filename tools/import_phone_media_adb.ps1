param(
    [string]$Destination = 'D:\images\PhoneImport',
    [string]$Adb = 'C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe',
    [string]$Serial = ''
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB was not found at '$Adb'." }
$deviceLines = @(& $Adb devices -l | Select-Object -Skip 1 | Where-Object { $_ -match '\sdevice(?:\s|$)' })
if (-not $Serial) {
    if ($deviceLines.Count -ne 1) { throw "Expected one authorized ADB device; found $($deviceLines.Count)." }
    $Serial = ($deviceLines[0] -split '\s+')[0]
}
if (-not ($deviceLines | Where-Object { $_ -match ('^' + [regex]::Escape($Serial) + '\s+device(?:\s|$)') })) {
    throw "ADB device '$Serial' is not connected and authorized."
}

$jobs = [System.Collections.Generic.List[object]]::new()
function Add-Pull([string]$remote, [string]$local) {
    $exists = @(& $Adb -s $Serial shell "if [ -d '$remote' ]; then echo yes; fi" 2>$null)
    if ($exists -contains 'yes') { $jobs.Add([pscustomobject]@{ Remote = $remote; Local = $local }) }
}

# Internal shared storage: camera roll and all WhatsApp media locations.
Add-Pull '/sdcard/DCIM' (Join-Path $Destination 'Internal storage')
Add-Pull '/sdcard/Pictures/WhatsApp' (Join-Path $Destination 'Internal storage\Pictures')
Add-Pull '/sdcard/WhatsApp' (Join-Path $Destination 'Internal storage')
Add-Pull '/sdcard/Android/media/com.whatsapp' (Join-Path $Destination 'Internal storage\Android\media')
Add-Pull '/sdcard/Android/media/com.whatsapp.w4b' (Join-Path $Destination 'Internal storage\Android\media')

# Removable storage UUIDs are exposed as directories under /storage.
$storageEntries = @(& $Adb -s $Serial shell 'ls -1 /storage' | ForEach-Object { $_.Trim() })
$sdCards = @($storageEntries | Where-Object { $_ -match '^[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}$' })
foreach ($uuid in $sdCards) {
    $remoteRoot = "/storage/$uuid"
    $localRoot = Join-Path $Destination 'SD card'
    Add-Pull "$remoteRoot/DCIM" $localRoot
    Add-Pull "$remoteRoot/Pictures/WhatsApp" (Join-Path $localRoot 'Pictures')
    Add-Pull "$remoteRoot/WhatsApp" $localRoot
    Add-Pull "$remoteRoot/Android/media/com.whatsapp" (Join-Path $localRoot 'Android\media')
    Add-Pull "$remoteRoot/Android/media/com.whatsapp.w4b" (Join-Path $localRoot 'Android\media')
}

if (-not $jobs.Count) { throw 'No DCIM or WhatsApp media folders were found on the phone.' }
Write-Output "ADB device $Serial; pulling $($jobs.Count) folders into $Destination. Source files will remain on the phone."
foreach ($job in $jobs) {
    New-Item -ItemType Directory -Path $job.Local -Force | Out-Null
    Write-Output "Pulling $($job.Remote)"
    & $Adb -s $Serial pull -a $job.Remote $job.Local
    if ($LASTEXITCODE -ne 0) { throw "ADB pull failed for '$($job.Remote)' with exit code $LASTEXITCODE." }
}

$files = @(Get-ChildItem -LiteralPath $Destination -File -Recurse -Force -ErrorAction SilentlyContinue)
$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
if ($null -eq $totalBytes) { $totalBytes = 0 }
Write-Output "ADB transfer complete: $($files.Count) files, $([Math]::Round($totalBytes / 1GB, 2)) GB under $Destination."
