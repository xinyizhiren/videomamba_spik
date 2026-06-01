$ErrorActionPreference = "Stop"

$source = "C:\Users\30810\Desktop\月度汇报\月度汇报_2026_05_整合版.pptx"
$out = "C:\Users\30810\Desktop\月度汇报\月度汇报_2026_05_整合版_结尾版.pptx"
$preview = "D:\code\PYTHON\video_sm\outputs\monthly_report\standard_ending_preview.png"

$msoTrue = -1
$msoFalse = 0
$ppLayoutBlank = 12
$msoShapeRectangle = 1
$msoTextOrientationHorizontal = 1
$ppAlignCenter = 2
$msoAnchorMiddle = 3

function RGBColor($r, $g, $b) {
    return $r + ($g * 256) + ($b * 65536)
}

function AddText($slide, $text, $x, $y, $w, $h, $size, $color, $bold = $false) {
    $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, [single]$x, [single]$y, [single]$w, [single]$h)
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.WordWrap = $msoTrue
    $shape.TextFrame.VerticalAnchor = $msoAnchorMiddle
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = [single]$size
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    if ($bold) { $shape.TextFrame.TextRange.Font.Bold = $msoTrue } else { $shape.TextFrame.TextRange.Font.Bold = $msoFalse }
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $ppAlignCenter
}

Copy-Item -LiteralPath $source -Destination $out -Force

$powerpoint = New-Object -ComObject PowerPoint.Application
$pres = $powerpoint.Presentations.Open($out, $msoFalse, $msoFalse, $msoFalse)

try {
    $sw = [single]$pres.PageSetup.SlideWidth
    $sh = [single]$pres.PageSetup.SlideHeight

    if ($pres.Slides.Count -gt 0) {
        $pres.Slides.Item($pres.Slides.Count).Delete()
    }

    $slide = $pres.Slides.Add($pres.Slides.Count + 1, $ppLayoutBlank)

    $dark = RGBColor 15 23 42
    $paper = RGBColor 255 255 255
    $muted = RGBColor 203 213 225
    $line = RGBColor 148 163 184

    $bg = $slide.Shapes.AddShape($msoShapeRectangle, 0, 0, $sw, $sh)
    $bg.Fill.ForeColor.RGB = $dark
    $bg.Line.Visible = $msoFalse
    $bg.ZOrder(1) | Out-Null

    AddText $slide "谢谢聆听" 0 ($sh * 0.30) $sw 70 40 $paper $true
    AddText $slide "请批评指正" 0 ($sh * 0.45) $sw 42 20 $muted $false

    $footerLine = $slide.Shapes.AddShape($msoShapeRectangle, 60, ($sh - 58), ($sw - 120), 1)
    $footerLine.Fill.ForeColor.RGB = $line
    $footerLine.Line.Visible = $msoFalse
    AddText $slide "Monthly Research 2026.05" 0 ($sh - 42) $sw 16 8 $line $false

    $pres.Save()
    $slide.Export($preview, "PNG", 1280, 720)
    Write-Output ("saved={0}" -f $out)
    Write-Output ("slides={0}" -f $pres.Slides.Count)
    Write-Output ("preview={0}" -f $preview)
}
finally {
    $pres.Close()
    $powerpoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint) | Out-Null
}
