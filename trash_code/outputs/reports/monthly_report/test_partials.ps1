$ErrorActionPreference = "Stop"

$dir = "D:\code\PYTHON\video_sm\outputs\monthly_report\partials"
$files = Get-ChildItem -Path $dir -Filter "*.pptx" | Sort-Object Name

$powerpoint = New-Object -ComObject PowerPoint.Application
try {
    foreach ($file in $files) {
        try {
            $presentation = $powerpoint.Presentations.Open($file.FullName, $true, $true, $false)
            $count = $presentation.Slides.Count
            $presentation.Close()
            Write-Output ("OK {0} slides={1}" -f $file.Name, $count)
        }
        catch {
            Write-Output ("FAIL {0} error={1}" -f $file.Name, $_.Exception.Message)
            break
        }
    }
}
finally {
    $powerpoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint) | Out-Null
}
