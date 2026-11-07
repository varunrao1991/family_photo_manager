param(
    [string]$Destination = '',
    [string]$PhoneName = ''
)

$ErrorActionPreference = 'Stop'

function Get-EnvConfig($key, $default = '') {
    $envFile = Join-Path $PSScriptRoot '..\.env'
    if (Test-Path -LiteralPath $envFile) {
        foreach ($line in Get-Content -LiteralPath $envFile) {
            $line = $line.Trim()
            if ($line -and -not $line.StartsWith('#') -and $line.StartsWith("$key=")) {
                return $line.Substring($key.Length + 1).Trim("`"' ")
            }
        }
    }
    return $default
}

if (-not $Destination) {
    $Destination = Get-EnvConfig 'PHONE_IMPORT_DESTINATION' 'D:\images\PhoneImport'
}
if (-not $PhoneName) {
    $PhoneName = Get-EnvConfig 'PHONE_NAME' ''
}

$shell = New-Object -ComObject Shell.Application
$thisPc = $shell.Namespace('shell:MyComputerFolder')

# Find connected phone
$phone = $null
if ($PhoneName) {
    $phone = $thisPc.Items() | Where-Object { $_.Name -ieq $PhoneName } | Select-Object -First 1
}
if (-not $phone) {
    $phone = $thisPc.Items() | Where-Object { $_.Type -eq 'Mobile Phone' -or $_.Name -match '(?i)phone|galaxy|android|pixel' } | Select-Object -First 1
}

if (-not $phone) {
    Write-Output "No phone connected in This PC. Skipping phone media sync."
    exit 0
}

Write-Output "Phone detected: $($phone.Name)"
Write-Output "Target folder:  $Destination"

$extensions = @(
    '.jpg','.jpeg','.png','.webp','.heic','.heif','.gif','.bmp','.tif','.tiff',
    '.dng','.avif','.mp4','.mov','.m4v','.3gp','.3g2','.mkv','.avi','.webm'
)

$phoneFolder = $phone.GetFolder
$storages = @($phoneFolder.Items() | Where-Object { $_.IsFolder })

if (-not $storages.Count) {
    Write-Output "No readable storage found on phone. Ensure phone is unlocked."
    exit 0
}

$totalChecked = 0
$totalSkipped = 0
$totalCopied = 0
$failedFiles = [System.Collections.Generic.List[string]]::new()

function Sync-FolderItems($remoteFolderItem, $localDir) {
    if (-not $remoteFolderItem) { return }
    $remoteFolder = $remoteFolderItem.GetFolder
    if (-not $remoteFolder) { return }

    # Ensure local directory exists
    if (-not (Test-Path -LiteralPath $localDir)) {
        New-Item -ItemType Directory -Path $localDir -Force | Out-Null
    }
    $localShellFolder = $shell.Namespace($localDir)

    $items = @($remoteFolder.Items())
    $filesToCopy = [System.Collections.Generic.List[object]]::new()

    foreach ($item in $items) {
        if ($item.IsFolder) {
            # Recursively sync subfolders
            $subLocal = Join-Path $localDir $item.Name
            Sync-FolderItems $item $subLocal
        } else {
            $ext = [IO.Path]::GetExtension($item.Name).ToLowerInvariant()
            if ($extensions -contains $ext) {
                $script:totalChecked++
                $localPath = Join-Path $localDir $item.Name
                if (Test-Path -LiteralPath $localPath) {
                    $script:totalSkipped++
                } else {
                    $filesToCopy.Add($item)
                }
            }
        }
    }

    # Copy missing files in this folder
    foreach ($file in $filesToCopy) {
        $destPath = Join-Path $localDir $file.Name
        Write-Output "  + Copying: $($file.Name)"
        try {
            $localShellFolder.CopyHere($file, 16 + 1024)
            # Wait for transfer
            $retries = 20
            while (-not (Test-Path -LiteralPath $destPath) -and $retries -gt 0) {
                Start-Sleep -Milliseconds 500
                $retries--
            }
            if (Test-Path -LiteralPath $destPath) {
                $script:totalCopied++
            } else {
                $script:failedFiles.Add($destPath)
            }
        } catch {
            $script:failedFiles.Add($destPath)
        }
    }
}

foreach ($storage in $storages) {
    $stName = $storage.Name
    $stFolder = $storage.GetFolder
    Write-Output "Scanning $stName..."

    # 1. DCIM
    $dcim = $stFolder.Items() | Where-Object { $_.Name -ieq 'DCIM' } | Select-Object -First 1
    if ($dcim) {
        Sync-FolderItems $dcim (Join-Path $Destination "$stName\DCIM")
    }

    # 2. Pictures
    $pictures = $stFolder.Items() | Where-Object { $_.Name -ieq 'Pictures' } | Select-Object -First 1
    if ($pictures) {
        Sync-FolderItems $pictures (Join-Path $Destination "$stName\Pictures")
    }

    # 3. WhatsApp (root or Android/media)
    $wa = $stFolder.Items() | Where-Object { $_.Name -ieq 'WhatsApp' } | Select-Object -First 1
    if ($wa) {
        Sync-FolderItems $wa (Join-Path $Destination "$stName\WhatsApp")
    }

    $android = $stFolder.Items() | Where-Object { $_.Name -ieq 'Android' } | Select-Object -First 1
    if ($android) {
        $media = $android.GetFolder.Items() | Where-Object { $_.Name -ieq 'media' } | Select-Object -First 1
        if ($media) {
            $waMedia = $media.GetFolder.Items() | Where-Object { $_.Name -match '(?i)com\.whatsapp' }
            foreach ($wm in $waMedia) {
                Sync-FolderItems $wm (Join-Path $Destination "$stName\Android\media\$($wm.Name)")
            }
        }
    }
}

Write-Output ""
Write-Output "=== Phone Sync Summary ==="
Write-Output "Total files checked:   $totalChecked"
Write-Output "Already backed up:     $totalSkipped (skipped)"
Write-Output "New files copied:      $totalCopied"
if ($failedFiles.Count -gt 0) {
    Write-Output "Failed transfers:      $($failedFiles.Count)"
}

