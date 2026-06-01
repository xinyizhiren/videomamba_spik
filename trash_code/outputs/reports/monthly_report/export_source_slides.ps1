$ErrorActionPreference = "Stop"

$source = "D:\code\PYTHON\video_sm\outputs\monthly_report\source_original.pptx"
$outDir = "D:\code\PYTHON\video_sm\outputs\monthly_report\source_slide_images"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$powerpoint = New-Object -ComObject PowerPoint.Application
$presentation = $powerpoint.Presentations.Open($source, $true, $true, $false)

try {
    for ($i = 1; $i -le $presentation.Slides.Count; $i++) {
        $path = Join-Path $outDir ("source_slide_{0:D2}.png" -f $i)
        $presentation.Slides.Item($i).Export($path, "PNG", 1920, 1080)
    }
}
finally {
    $presentation.Close()
    $powerpoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint) | Out-Null
}

Write-Output "exported=$outDir"
