param(
    [string]$Destination = 'D:\images\PhoneImport',
    [string]$PhoneName = "varun's A16",
    [int]$QuietSeconds = 180,
    [int]$TimeoutHours = 4
)

$ErrorActionPreference = 'Stop'

function Find-Folder($folder, [string]$name) {
    $folder.Items() | Where-Object { $_.IsFolder -and $_.Name -ieq $name } | Select-Object -First 1
}

$shell = New-Object -ComObject Shell.Application
$thisPc = $shell.Namespace('shell:MyComputerFolder')
$phone = $thisPc.Items() | Where-Object { $_.Name -eq $PhoneName } | Select-Object -First 1
if (-not $phone) { throw "Connected phone '$PhoneName' was not found in This PC." }

$storages = @($phone.GetFolder.Items() | Where-Object { $_.IsFolder -and $_.Name -in @('Internal storage', 'SD card') })
if (-not $storages.Count) { throw 'The phone did not expose internal storage or an SD card.' }

$operations = [System.Collections.Generic.List[object]]::new()
foreach ($storage in $storages) {
    $storageFolder = $storage.GetFolder
    foreach ($name in @('DCIM', 'WhatsApp')) {
        $item = Find-Folder $storageFolder $name
        if ($item) { $operations.Add([pscustomobject]@{ Item = $item; RelativeParent = $storage.Name; Label = "$($storage.Name)\$name" }) }
    }

    $pictures = Find-Folder $storageFolder 'Pictures'
    if ($pictures) {
        $whatsAppPictures = Find-Folder $pictures.GetFolder 'WhatsApp'
        if ($whatsAppPictures) { $operations.Add([pscustomobject]@{ Item = $whatsAppPictures; RelativeParent = "$($storage.Name)\Pictures"; Label = "$($storage.Name)\Pictures\WhatsApp" }) }
    }

    $android = Find-Folder $storageFolder 'Android'
    if ($android) {
        $media = Find-Folder $android.GetFolder 'media'
        if ($media) {
            foreach ($appFolder in $media.GetFolder.Items() | Where-Object { $_.IsFolder -and $_.Name -match '(?i)com\.whatsapp' }) {
                $operations.Add([pscustomobject]@{ Item = $appFolder; RelativeParent = "$($storage.Name)\Android\media"; Label = "$($storage.Name)\Android\media\$($appFolder.Name)" })
            }
        }
    }
}

if (-not $operations.Count) { throw 'No DCIM or WhatsApp folders were found on the phone.' }
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
Write-Output "Copying $($operations.Count) camera/WhatsApp folders into $Destination. Phone originals will remain in place."

foreach ($operation in $operations) {
    $targetParent = Join-Path $Destination $operation.RelativeParent
    New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    $targetPath = Join-Path $targetParent $operation.Item.Name
    if (Test-Path -LiteralPath $targetPath) {
        Write-Output "Already present, leaving existing copy in place: $($operation.Label)"
        continue
    }
    $targetFolder = $shell.Namespace($targetParent)
    if (-not $targetFolder) { throw "Could not open destination '$targetParent'." }
    Write-Output "Starting $($operation.Label)"
    $targetFolder.CopyHere($operation.Item, 16)
}

# MTP folder copies run asynchronously. Wait until the local tree's file count and
# total bytes have remained unchanged for a quiet period before reporting success.
$startedAt = [DateTime]::UtcNow
$lastChangeAt = $startedAt
$previousSignature = ''
$lastReportAt = [DateTime]::UtcNow
while ([DateTime]::UtcNow -lt $startedAt.AddHours($TimeoutHours)) {
    Start-Sleep -Seconds 5
    $files = @(Get-ChildItem -LiteralPath $Destination -File -Recurse -Force -ErrorAction SilentlyContinue)
    $totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $totalBytes) { $totalBytes = 0 }
    $signature = "$($files.Count):$totalBytes"
    if ($signature -ne $previousSignature) {
        $previousSignature = $signature
        $lastChangeAt = [DateTime]::UtcNow
    }
    if (([DateTime]::UtcNow - $lastReportAt).TotalSeconds -ge 30) {
        Write-Output "Transfer active: $($files.Count) files, $([Math]::Round($totalBytes / 1GB, 2)) GB copied so far."
        $lastReportAt = [DateTime]::UtcNow
    }
    if (([DateTime]::UtcNow - $lastChangeAt).TotalSeconds -ge $QuietSeconds -and $files.Count -gt 0) {
        Write-Output "Transfer settled: $($files.Count) files, $([Math]::Round($totalBytes / 1GB, 2)) GB under $Destination."
        exit 0
    }
}

throw "Transfer did not settle within $TimeoutHours hours. Files copied so far remain in $Destination; phone originals were not deleted."
