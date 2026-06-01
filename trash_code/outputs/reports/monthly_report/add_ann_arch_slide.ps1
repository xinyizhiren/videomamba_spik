$ErrorActionPreference = "Stop"

$deckPaths = @(
    "C:\Users\30810\Desktop\月度汇报\月度汇报_2026_05_整合版.pptx",
    "C:\Users\30810\Desktop\月度汇报\月度汇报_2026_05_可编辑重做版.pptx"
)
$previewDir = "D:\code\PYTHON\video_sm\outputs\monthly_report\ann_arch_preview"
New-Item -ItemType Directory -Force -Path $previewDir | Out-Null

$msoTrue = -1
$msoFalse = 0
$ppLayoutBlank = 12
$msoShapeRectangle = 1
$msoShapeRoundedRectangle = 5
$msoTextOrientationHorizontal = 1
$ppAlignLeft = 1
$ppAlignCenter = 2
$msoAnchorTop = 1
$msoAnchorMiddle = 3

function RGBColor($r, $g, $b) {
    return $r + ($g * 256) + ($b * 65536)
}

$C = @{
    Dark2 = RGBColor 30 41 59
    Ink = RGBColor 24 31 42
    Muted = RGBColor 100 116 139
    Paper = RGBColor 255 255 255
    Soft = RGBColor 246 248 251
    Line = RGBColor 220 226 235
    Blue = RGBColor 37 99 235
    Cyan = RGBColor 8 145 178
    Green = RGBColor 22 163 74
    Amber = RGBColor 217 119 6
    Red = RGBColor 220 38 38
    Violet = RGBColor 124 58 237
}

function Add-TextBox($slide, $text, $x, $y, $w, $h, $size, $color, $bold = $false, $align = 1) {
    try {
        $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, [single]$x, [single]$y, [single]$w, [single]$h)
    }
    catch {
        Write-Output ("Add-TextBox failed: x={0} y={1} w={2} h={3} text={4}" -f $x, $y, $w, $h, $text)
        throw
    }
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.WordWrap = $msoTrue
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.VerticalAnchor = $msoAnchorTop
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = [single]$size
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    if ($bold) { $shape.TextFrame.TextRange.Font.Bold = $msoTrue } else { $shape.TextFrame.TextRange.Font.Bold = $msoFalse }
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Pill($slide, $text, $x, $y, $w, $h, $fill, $size = 10, $textColor = $null) {
    if ($null -eq $textColor) { $textColor = $C.Paper }
    $shape = $slide.Shapes.AddShape($msoShapeRoundedRectangle, [single]$x, [single]$y, [single]$w, [single]$h)
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.Visible = $msoFalse
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.WordWrap = $msoFalse
    $shape.TextFrame.VerticalAnchor = $msoAnchorMiddle
    $shape.TextFrame.MarginLeft = 4
    $shape.TextFrame.MarginRight = 4
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = [single]$size
    $shape.TextFrame.TextRange.Font.Bold = $msoTrue
    $shape.TextFrame.TextRange.Font.Color.RGB = $textColor
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $ppAlignCenter
    return $shape
}

function Add-Box($slide, $x, $y, $w, $h, $title, $body, $accent) {
    $box = $slide.Shapes.AddShape($msoShapeRoundedRectangle, [single]$x, [single]$y, [single]$w, [single]$h)
    $box.Fill.ForeColor.RGB = $C.Paper
    $box.Line.ForeColor.RGB = $accent
    $box.Line.Weight = 1.2
    Add-TextBox $slide $title ($x + 8) ($y + 10) ($w - 16) 18 10.5 $accent $true $ppAlignCenter | Out-Null
    Add-TextBox $slide $body ($x + 8) ($y + 34) ($w - 16) ($h - 40) 8.5 $C.Muted $false $ppAlignCenter | Out-Null
}

function Add-Line($slide, $x, $y, $w, $color) {
    $line = $slide.Shapes.AddShape($msoShapeRectangle, [single]$x, [single]$y, [single]$w, 2)
    $line.Fill.ForeColor.RGB = $color
    $line.Line.Visible = $msoFalse
}

function Add-Card($slide, $x, $y, $w, $h, $title, $body, $accent) {
    $box = $slide.Shapes.AddShape($msoShapeRoundedRectangle, [single]$x, [single]$y, [single]$w, [single]$h)
    $box.Fill.ForeColor.RGB = $C.Paper
    $box.Line.ForeColor.RGB = $C.Line
    $box.Line.Weight = 1
    $bar = $slide.Shapes.AddShape($msoShapeRectangle, [single]$x, [single]$y, 4, [single]$h)
    $bar.Fill.ForeColor.RGB = $accent
    $bar.Line.Visible = $msoFalse
    Add-TextBox $slide $title ($x + 16) ($y + 12) ($w - 32) 20 11 $C.Ink $true $ppAlignLeft | Out-Null
    Add-TextBox $slide $body ($x + 16) ($y + 39) ($w - 32) ($h - 45) 8.8 $C.Muted $false $ppAlignLeft | Out-Null
}

function Add-Footer($slide, $text = "Monthly Research 2026.05") {
    $line = $slide.Shapes.AddShape($msoShapeRectangle, 45, 502, 870, 1)
    $line.Fill.ForeColor.RGB = $C.Line
    $line.Line.Visible = $msoFalse
    Add-TextBox $slide $text 45 512 350 14 7.5 (RGBColor 145 155 171) $false $ppAlignLeft | Out-Null
}

function Slide-ContainsTitle($slide, $needle) {
    foreach ($shape in $slide.Shapes) {
        try {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                if ($shape.TextFrame.TextRange.Text -like "*$needle*") { return $true }
            }
        } catch {}
    }
    return $false
}

function Add-AnnSlide($pres) {
    $title = "原始 ANN VideoMamba 模型结构"

    for ($i = $pres.Slides.Count; $i -ge 1; $i--) {
        if (Slide-ContainsTitle $pres.Slides.Item($i) $title) {
            $pres.Slides.Item($i).Delete()
        }
    }

    $insertIndex = 11
    if ($insertIndex -gt ($pres.Slides.Count + 1)) { $insertIndex = $pres.Slides.Count + 1 }
    $slide = $pres.Slides.Add($insertIndex, $ppLayoutBlank)

    $bg = $slide.Shapes.AddShape($msoShapeRectangle, 0, 0, 960, 540)
    $bg.Fill.ForeColor.RGB = $C.Paper
    $bg.Line.Visible = $msoFalse
    $bg.ZOrder(1) | Out-Null

    Add-Pill $slide "ANN baseline" 45 28 122 26 $C.Dark2 9.5 | Out-Null
    Add-TextBox $slide $title 45 86 760 34 24 $C.Ink $true $ppAlignLeft | Out-Null
    Add-TextBox $slide "clean VideoMamba small：全浮点激活，无 LIF / spike 层，无 SNN 时间步重复；作为后续 SNN 训练和蒸馏的 teacher。" 45 128 830 28 11.5 $C.Muted $false $ppAlignLeft | Out-Null

    $items = @(
        @("Input video", "B x 3 x 16 x 224 x 224", $C.Blue, 45, 210, 112),
        @("PatchEmbed", "Conv3D patchify`n384 x 16 x 14 x 14", $C.Cyan, 180, 210, 122),
        @("Tokens + PE", "3136 patches + CLS`nB x 3137 x 384", $C.Violet, 325, 210, 126),
        @("24 x Blocks", "VideoMamba block`nResidual + LN + Mamba", $C.Green, 475, 200, 145),
        @("Final Norm", "residual add`nLayerNorm", $C.Amber, 645, 210, 112),
        @("Head", "MeanPool`nLinear 384 -> 12", $C.Red, 780, 210, 118)
    )

    for ($i = 0; $i -lt $items.Count; $i++) {
        $item = $items[$i]
        $boxTitle = $item[0]
        $boxBody = $item[1]
        $boxColor = $item[2]
        $boxX = [double]$item[3]
        $boxY = [double]$item[4]
        $boxW = [double]$item[5]
        Add-Box $slide $boxX $boxY $boxW 92 $boxTitle $boxBody $boxColor
        if ($i -lt ($items.Count - 1)) {
            $nextItem = $items[$i + 1]
            $x1 = $boxX + $boxW + 10
            $x2 = [double]$nextItem[3] - 10
            Add-Line $slide $x1 255 ($x2 - $x1) $C.Line
        }
    }

    Add-Card $slide 45 350 270 95 "双视角训练 / 推理" "训练时 view1 与 view2 共享同一个 ANN 权重，分别 forward 后平均 logits；验证和测试时可单 view 输入得到分类 logits。" $C.Blue
    Add-Card $slide 345 350 270 95 "Block 内部不做脉冲化" "原始 ANN 的 LayerNorm、Mamba mixer、residual 分支均保持浮点连续激活，没有二值化或膜电位状态。" $C.Green
    Add-Card $slide 645 350 270 95 "后续 SNN 的参照系" "SNN 实验从这个 ANN best.pth 加载权重，并用 clean ANN teacher 做 distillation，目标是在脉冲化后恢复精度。" $C.Cyan

    Add-Footer $slide
    return $slide
}

$powerpoint = New-Object -ComObject PowerPoint.Application

try {
    foreach ($deck in $deckPaths) {
        if (-not (Test-Path -LiteralPath $deck)) {
            Write-Output "skip missing: $deck"
            continue
        }
        $pres = $powerpoint.Presentations.Open($deck, $msoFalse, $msoFalse, $msoFalse)
        try {
            $slide = Add-AnnSlide $pres
            $pres.Save()
            $name = [IO.Path]::GetFileNameWithoutExtension($deck)
            $slide.Export((Join-Path $previewDir ($name + "_ann_arch.png")), "PNG", 1280, 720)
            Write-Output ("updated={0}; slides={1}" -f $deck, $pres.Slides.Count)
        }
        finally {
            $pres.Close()
        }
    }
}
finally {
    $powerpoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint) | Out-Null
}
