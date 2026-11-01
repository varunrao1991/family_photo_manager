param(
    [string]$Destination = 'D:\images\PhoneImport',
    [string]$Adb = 'C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe',
    [string]$Serial = ''
)

$ErrorActionPreference = 'Stop'
$extensions = @('.jpg','.jpeg','.png','.webp','.heic','.heif','.gif','.bmp','.tif','.tiff','.dng','.avif','.mp4','.mov','.m4v','.3gp','.3g2','.mkv','.avi','.webm','.mpg','.mpeg','.wmv','.mts','.m2ts','.ts','.flv')
if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB was not found at '$Adb'." }
$devices = @(& $Adb devices -l | Select-Object -Skip 1 | Where-Object { $_ -match '\sdevice(?:\s|$)' })
if (-not $Serial) {
    if ($devices.Count -ne 1) { throw "Expected one authorized ADB device; found $($devices.Count)." }
    $Serial = ($devices[0] -split '\s+')[0]
}

$skipped = [System.Collections.Generic.List[string]]::new()
$transferErrors = [System.Collections.Generic.List[string]]::new()
$checked = 0
$pulled = 0

function Copy-MediaTree([string]$remoteRoot, [string]$localRoot) {
    & $Adb -s $Serial shell "test -d '$remoteRoot'"
    if ($LASTEXITCODE -ne 0) { return }
    Write-Output "Checking $remoteRoot"
    $records = @(& $Adb -s $Serial shell "find '$remoteRoot' -type f -printf '%p\t%s\n'")
    if ($LASTEXITCODE -ne 0) { Write-Output "Could not enumerate $remoteRoot"; return }
    foreach ($record in $records) {
        $parts = $record.TrimEnd("`r") -split "`t", 2
        if ($parts.Count -ne 2 -or $parts[1] -notmatch '^\d+$') { continue }
        $remoteFile = $parts[0]
        $expectedLength = [int64]$parts[1]
        if ($extensions -notcontains [IO.Path]::GetExtension($remoteFile).ToLowerInvariant()) { continue }
        $relative = $remoteFile.Substring($remoteRoot.Length).TrimStart('/') -replace '/', '\'
        $localFile = Join-Path $localRoot $relative
        $script:checked++
        if (($script:checked % 250) -eq 0) { Write-Output "Checked $script:checked media files; pulled $script:pulled missing files so far." }
        if ((Test-Path -LiteralPath $localFile) -and (Get-Item -LiteralPath $localFile).Length -eq $expectedLength) { continue }
        if ($remoteFile -match '20251021_135315\.mp4$') {
            $skipped.Add("$remoteFile (repeated remote I/O error at 51%)")
            continue
        }
        $localDirectory = Split-Path -Parent $localFile
        New-Item -ItemType Directory -Path $localDirectory -Force | Out-Null
        $output = @(& $Adb -s $Serial pull -a $remoteFile $localDirectory 2>&1)
        if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $localFile) -and (Get-Item -LiteralPath $localFile).Length -eq $expectedLength) {
            $script:pulled++
        } else {
            $transferErrors.Add($remoteFile)
            Write-Output "Could not copy $remoteFile"
        }
    }
}

$internal = Join-Path $Destination 'Internal storage'
Copy-MediaTree '/sdcard/DCIM' (Join-Path $internal 'DCIM')
Copy-MediaTree '/sdcard/Pictures/WhatsApp' (Join-Path $internal 'Pictures\WhatsApp')
Copy-MediaTree '/sdcard/WhatsApp' (Join-Path $internal 'WhatsApp')
Copy-MediaTree '/sdcard/Android/media/com.whatsapp' (Join-Path $internal 'Android\media\com.whatsapp')
Copy-MediaTree '/sdcard/Android/media/com.whatsapp.w4b' (Join-Path $internal 'Android\media\com.whatsapp.w4b')

$storageEntries = @(& $Adb -s $Serial shell 'ls -1 /storage' | ForEach-Object { $_.Trim() })
$sdCards = @($storageEntries | Where-Object { $_ -match '^[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}$' })
foreach ($uuid in $sdCards) {
    $remoteRoot = "/storage/$uuid"
    $sd = Join-Path $Destination 'SD card'
    Copy-MediaTree "$remoteRoot/DCIM" (Join-Path $sd 'DCIM')
    Copy-MediaTree "$remoteRoot/Pictures/WhatsApp" (Join-Path $sd 'Pictures\WhatsApp')
    Copy-MediaTree "$remoteRoot/WhatsApp" (Join-Path $sd 'WhatsApp')
    Copy-MediaTree "$remoteRoot/Android/media/com.whatsapp" (Join-Path $sd 'Android\media\com.whatsapp')
    Copy-MediaTree "$remoteRoot/Android/media/com.whatsapp.w4b" (Join-Path $sd 'Android\media\com.whatsapp.w4b')
}

Write-Output "Checked $checked image/video files; pulled $pulled missing files."
if ($skipped.Count) { Write-Output "Skipped $($skipped.Count) unreadable source file(s):"; $skipped | ForEach-Object { Write-Output "  $_" } }
if ($transferErrors.Count) { Write-Output "Could not copy $($transferErrors.Count) other file(s):"; $transferErrors | ForEach-Object { Write-Output "  $_" } }
