$ErrorActionPreference = "Stop"

$source = "D:\code\PYTHON\video_sm\outputs\monthly_report\monthly_report_2026_05_image_stable.pptx"
$outDir = "D:\code\PYTHON\video_sm\outputs\monthly_report\image_deck_preview"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$powerpoint = New-Object -ComObject PowerPoint.Application
$presentation = $powerpoint.Presentations.Open($source, $true, $true, $false)

try {
    $count = $presentation.Slides.Count
    for ($i = 1; $i -le $count; $i++) {
        $path = Join-Path $outDir ("preview_{0:D2}.png" -f $i)
        $presentation.Slides.Item($i).Export($path, "PNG", 1280, 720)
    }
    Write-Output "opened_slides=$count"
}
finally {
    $presentation.Close()
    $powerpoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint) | Out-Null
}
