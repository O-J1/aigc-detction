param(
  [int]$TrainRealCount = 64,
  [int]$TrainFakeCount = 64,
  [int]$ValRealCount = 16,
  [int]$ValFakeCount = 16
)

mkdir manifests -Force

$realRoots = @(
  [pscustomobject]@{
    Path = "E:\datasets\sampled_images\wikiarts"
    KeepPercent = 100
    MinPerLeafFolder = 1
  },
  [pscustomobject]@{
    Path = "E:\datasets\sampled_images\random_tagged_dataset_smol\random_tagged_dataset_smol"
    KeepPercent = 100
    MinPerLeafFolder = 1
  },
  [pscustomobject]@{
    Path = "E:\datasets\unsplash-research-dataset-lite-latest\photos"
    KeepPercent = 25
    MinPerLeafFolder = 1
  },
   [pscustomobject]@{
    Path = "D:\dataset"
    KeepPercent = 25
    MinPerLeafFolder = 1
  },
  [pscustomobject]@{
    Path = "D:\aigc-dataset\shard_5\shard_5\real"
    KeepPercent = 25
    MinPerLeafFolder = 1
  }

)

$fakeRoots = @(
  [pscustomobject]@{
    Path = "D:\aigc-dataset\images"
    KeepPercent = 50
    MinPerLeafFolder = 1
  },
  [pscustomobject]@{
    Path = "D:\aigc-dataset\shard_5\shard_5\fake"
    KeepPercent = 100
    MinPerLeafFolder = 1
  }
)

$exts = @("*.jpg", "*.jpeg", "*.png", "*.webp")

function Get-ImagesFromRoot($rootConfig) {
  $root = $rootConfig.Path
  $keepPercent = [double]$rootConfig.KeepPercent
  $minPerLeafFolder = [int]$rootConfig.MinPerLeafFolder

  if (-not (Test-Path $root)) {
    Write-Warning "Skipping missing root: $root"
    return @()
  }

  $files = foreach ($ext in $exts) {
    Get-ChildItem $root -Recurse -File -Filter $ext -ErrorAction SilentlyContinue
  }

  $keepFraction = [math]::Max(0.0, [math]::Min(1.0, $keepPercent / 100.0))
  if ($keepFraction -le 0.0) {
    return @()
  }

  $sampledFiles = $files |
    Group-Object DirectoryName |
    ForEach-Object {
      $folderFiles = @($_.Group)
      $sampleCount = [int][math]::Ceiling($folderFiles.Count * $keepFraction)
      $sampleCount = [math]::Max($minPerLeafFolder, $sampleCount)
      $sampleCount = [math]::Min($folderFiles.Count, $sampleCount)
      $folderFiles | Sort-Object { Get-Random } | Select-Object -First $sampleCount
    }

  return @($sampledFiles)
}

function Get-ImagePool($rootConfigs) {
  $files = foreach ($rootConfig in $rootConfigs) {
    Get-ImagesFromRoot $rootConfig
  }

  $files | Sort-Object { Get-Random }
}

function Show-SamplingSummary($name, $rootConfigs) {
  foreach ($rootConfig in $rootConfigs) {
    $root = $rootConfig.Path
    $total = 0
    foreach ($ext in $exts) {
      $total += @(Get-ChildItem $root -Recurse -File -Filter $ext -ErrorAction SilentlyContinue).Count
    }
    $sampled = @(Get-ImagesFromRoot $rootConfig).Count
    Write-Host "$name root: $root"
    Write-Host "  keep=$($rootConfig.KeepPercent)% sampled_after_leaf_strata=$sampled total=$total"
  }
}

function Write-Manifest($path, $realFiles, $fakeFiles, $split) {
  $rows = @()
  $rows += $realFiles | ForEach-Object {
    [pscustomobject]@{
      path = $_.FullName
      label = 0
      split = $split
      source_tier = "local_real"
      source_dataset = "local_smoke"
      generator = ""
      task_type = ""
      width = ""
      height = ""
      sha256 = ""
    }
  }
  $rows += $fakeFiles | ForEach-Object {
    [pscustomobject]@{
      path = $_.FullName
      label = 1
      split = $split
      source_tier = "local_aigc"
      source_dataset = "local_smoke"
      generator = "mixed_local_aigc"
      task_type = ""
      width = ""
      height = ""
      sha256 = ""
    }
  }
  $rows | Export-Csv $path -NoTypeInformation
  Write-Host "Wrote $($rows.Count) rows to $path"
}

Show-SamplingSummary "real" $realRoots
Show-SamplingSummary "fake" $fakeRoots

$realPool = @(Get-ImagePool $realRoots)
$fakePool = @(Get-ImagePool $fakeRoots)

$trainRealCount = $TrainRealCount
$trainFakeCount = $TrainFakeCount
$valRealCount = $ValRealCount
$valFakeCount = $ValFakeCount

$trainReal = @($realPool | Select-Object -First $trainRealCount)
$valReal = @($realPool | Select-Object -Skip $trainRealCount -First $valRealCount)
$trainFake = @($fakePool | Select-Object -First $trainFakeCount)
$valFake = @($fakePool | Select-Object -Skip $trainFakeCount -First $valFakeCount)

Write-Host "Selected real train=$($trainReal.Count) val=$($valReal.Count) from pool=$($realPool.Count)"
Write-Host "Selected fake train=$($trainFake.Count) val=$($valFake.Count) from pool=$($fakePool.Count)"

Write-Manifest "manifests\smoke_train.csv" $trainReal $trainFake "train"
Write-Manifest "manifests\smoke_val.csv" $valReal $valFake "val"