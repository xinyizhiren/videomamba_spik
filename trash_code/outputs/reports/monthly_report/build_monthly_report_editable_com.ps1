$ErrorActionPreference = "Stop"

$source = "D:\code\PYTHON\video_sm\outputs\monthly_report\source_original.pptx"
$out = "D:\code\PYTHON\video_sm\outputs\monthly_report\monthly_report_2026_05_editable_com.pptx"
$previewDir = "D:\code\PYTHON\video_sm\outputs\monthly_report\editable_com_preview"

New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
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
    Dark = RGBColor 15 23 42
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
    Slate = RGBColor 203 213 225
    LightText = RGBColor 226 232 240
}

function Add-BlankSlide($pres, $bgColor) {
    $slide = $pres.Slides.Add($pres.Slides.Count + 1, $ppLayoutBlank)
    $bg = $slide.Shapes.AddShape($msoShapeRectangle, 0, 0, 960, 540)
    $bg.Fill.ForeColor.RGB = $bgColor
    $bg.Line.Visible = $msoFalse
    $bg.ZOrder(1) | Out-Null
    return $slide
}

function Add-TextBox($slide, $text, $x, $y, $w, $h, $size, $color, $bold = $false, $align = 1) {
    $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $x, $y, $w, $h)
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

function Add-Pill($slide, $text, $x, $y, $w, $h, $fill, $size = 12, $textColor = $null) {
    if ($null -eq $textColor) { $textColor = $C.Paper }
    $shape = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, $w, $h)
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

function Add-Card($slide, $x, $y, $w, $h, $title, $body, $accent) {
    $box = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, $w, $h)
    $box.Fill.ForeColor.RGB = $C.Paper
    $box.Line.ForeColor.RGB = $C.Line
    $box.Line.Weight = 1
    $bar = $slide.Shapes.AddShape($msoShapeRectangle, $x, $y, 4, $h)
    $bar.Fill.ForeColor.RGB = $accent
    $bar.Line.Visible = $msoFalse
    Add-TextBox $slide $title ($x + 18) ($y + 16) ($w - 36) 24 15 $C.Ink $true $ppAlignLeft | Out-Null
    $bodyH = $h - 58
    if ($bodyH -lt 16) { $bodyH = 16 }
    Add-TextBox $slide $body ($x + 18) ($y + 48) ($w - 36) $bodyH 10.5 $C.Muted $false $ppAlignLeft | Out-Null
}

function Add-Footer($slide, $text = "Monthly Research 2026.05") {
    $line = $slide.Shapes.AddShape($msoShapeRectangle, 45, 502, 870, 1)
    $line.Fill.ForeColor.RGB = $C.Line
    $line.Line.Visible = $msoFalse
    Add-TextBox $slide $text 45 512 350 14 7.5 (RGBColor 145 155 171) $false $ppAlignLeft | Out-Null
}

function Add-Metric($slide, $value, $label, $x, $y, $color) {
    $box = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, 165, 62)
    $box.Fill.ForeColor.RGB = $C.Dark2
    $box.Line.ForeColor.RGB = RGBColor 58 73 94
    $box.Line.Weight = 1
    Add-TextBox $slide $value ($x + 14) ($y + 8) 135 30 28 $color $true $ppAlignLeft | Out-Null
    Add-TextBox $slide $label ($x + 14) ($y + 38) 140 18 10 $C.Slate $false $ppAlignLeft | Out-Null
}

function Add-SectionLabel($slide, $label) {
    Add-Pill $slide $label 45 28 120 26 $C.Dark2 9.5 | Out-Null
}

function Build-Cover($pres) {
    $slide = Add-BlankSlide $pres $C.Dark
    Add-TextBox $slide "月度研究汇报" 45 42 380 36 26 $C.Slate $true | Out-Null
    Add-TextBox $slide "SpikingFormer 创新点" 45 110 690 50 40 $C.Paper $true | Out-Null
    Add-TextBox $slide "与 VideoMamba 脉冲化项目进展" 45 170 760 54 40 $C.Paper $true | Out-Null
    Add-TextBox $slide "2026.05｜阶段性实验总结与下一步计划" 45 250 650 28 20 $C.Slate $false | Out-Null
    Add-Pill $slide "第一部分：已有创新点 PPT" 45 350 235 34 $C.Blue 14 | Out-Null
    Add-Pill $slide "第二部分：VideoMamba SNN 项目" 300 350 260 34 $C.Cyan 14 | Out-Null
    Add-Metric $slide "94.12%" "ANN val baseline" 745 70 $C.Green
    Add-Metric $slide "83.09%" "24-block LIF SNN best" 745 160 $C.Cyan
    Add-Metric $slide "24" "active LIF spike layers" 745 250 $C.Violet
    Add-Footer $slide "Monthly Research 2026.05｜SNN Video Understanding"
}

function Build-Agenda($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "汇报结构"
    Add-TextBox $slide "本月工作整理为两个相互衔接的创新点" 45 88 760 36 24 $C.Ink $true | Out-Null
    Add-TextBox $slide "前半部分保留已有 PPT 内容；后半部分补充当前 VideoMamba 脉冲化项目的路线、失败尝试、实验结果和下一步计划。" 45 132 820 36 12 $C.Muted $false | Out-Null
    Add-Card $slide 45 220 395 150 "Part I｜SpikingFormer 创新点" "基于粗粒度理解的脉冲信息筛选机制探索：创新动机、整体流程、局部筛选机制，以及失败尝试如何帮助定位问题。" $C.Blue
    Add-Card $slide 520 220 395 150 "Part II｜VideoMamba SNN 项目" "以 clean VideoMamba ANN 为起点，比较 ANN2SNN 转换、自定义 signed spike、SpikingJelly LIF 三条路线，形成可训练 unsigned LIF 主线。" $C.Cyan
    Add-Footer $slide
}

function Build-PartOne($pres) {
    $slide = Add-BlankSlide $pres $C.Dark
    Add-TextBox $slide "Part I" 50 70 180 32 22 (RGBColor 148 163 184) $true | Out-Null
    Add-TextBox $slide "基于粗粒度理解的" 50 155 620 48 34 $C.Paper $true | Out-Null
    Add-TextBox $slide "脉冲信息筛选机制探索" 50 210 680 48 34 $C.Paper $true | Out-Null
    Add-TextBox $slide "以下 5 页为原 PPT 内容，直接由 PowerPoint 插入，保持原始可编辑结构。" 50 300 760 26 16 $C.Slate $false | Out-Null
    Add-Pill $slide "创新动机｜整体流程｜局部筛选机制｜失败尝试的作用" 50 380 510 34 $C.Blue 13 | Out-Null
    Add-Footer $slide "Part I｜SpikingFormer Innovation"
}

function Build-PartTwo($pres) {
    $slide = Add-BlankSlide $pres $C.Dark
    Add-TextBox $slide "Part II" 50 70 180 32 22 (RGBColor 148 163 184) $true | Out-Null
    Add-TextBox $slide "VideoMamba 脉冲化项目进展" 50 165 720 52 36 $C.Paper $true | Out-Null
    Add-TextBox $slide "从直接转换失败，到基于 ANN 权重的可训练 SNN：当前已完成 full 24-block unsigned LIF 的可行性验证。" 50 245 760 34 16 $C.Slate $false | Out-Null
    Add-Pill $slide "ANN baseline 94.12%" 50 360 210 30 $C.Green 11 | Out-Null
    Add-Pill $slide "ANN2SNN direct conversion failed" 285 360 275 30 $C.Red 11 | Out-Null
    Add-Pill $slide "24-block unsigned LIF best 83.09%" 585 360 300 30 $C.Cyan 11 | Out-Null
    Add-Footer $slide "Part II｜VideoMamba SNN"
}

function Build-ProjectFrame($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "项目定位"
    Add-TextBox $slide "问题不是「把层替换掉」，而是让 VideoMamba 的表示逐步适应脉冲传输" 45 88 830 34 21 $C.Ink $true | Out-Null
    Add-Card $slide 45 200 260 175 "约束" "数据集较小，不能从随机初始化学到足够强的时空表示；必须复用 clean ANN 预训练权重。" $C.Amber
    Add-Card $slide 350 200 260 175 "目标" "在 Mamba block 外部引入脉冲层，逐步提高脉冲化程度，并确认传输数据尽可能为 {0,1}。" $C.Cyan
    Add-Card $slide 655 200 260 175 "策略" "用 teacher distillation 与分阶段扩展 block 范围，缓解一次性插入大量 LIF 导致的精度崩溃。" $C.Green
    Add-Footer $slide
}

function Build-ConversionFail($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "失败路线"
    Add-TextBox $slide "直接 ANN2SNN 转换不适配 VideoMamba" 45 86 780 34 24 $C.Ink $true | Out-Null
    Add-TextBox $slide "加载 ANN 权重、插入 spike 层并做阈值校准，但不重新训练。随着脉冲 block 增多，精度快速坍塌。" 45 130 820 28 11.5 $C.Muted $false | Out-Null
    $headers = @("run","blocks","ANN val","SNN val","drop","test")
    $rows = @(
        @("block0","0","94.12","74.26","-19.85","72.67"),
        @("block01","0,1","94.12","67.65","-26.47","66.15"),
        @("block0123","0..3","94.12","51.47","-42.65","55.28")
    )
    $x0 = 45; $y0 = 200; $cw = 72; $rh = 28
    for ($i=0; $i -lt $headers.Count; $i++) {
        $cell = $slide.Shapes.AddShape($msoShapeRectangle, $x0 + $i*$cw, $y0, $cw, $rh)
        $cell.Fill.ForeColor.RGB = $C.Red; $cell.Line.ForeColor.RGB = $C.Line
        Add-TextBox $slide $headers[$i] ($x0 + $i*$cw + 4) ($y0 + 7) ($cw - 8) 14 8.5 $C.Paper $true $ppAlignCenter | Out-Null
    }
    for ($r=0; $r -lt $rows.Count; $r++) {
        for ($i=0; $i -lt $headers.Count; $i++) {
            $cell = $slide.Shapes.AddShape($msoShapeRectangle, $x0 + $i*$cw, $y0 + ($r+1)*$rh, $cw, $rh)
            $cell.Fill.ForeColor.RGB = $(if ($r % 2 -eq 0) { $C.Paper } else { $C.Soft })
            $cell.Line.ForeColor.RGB = $C.Line
            Add-TextBox $slide $rows[$r][$i] ($x0 + $i*$cw + 4) ($y0 + ($r+1)*$rh + 7) ($cw - 8) 14 8.5 $C.Ink $false $ppAlignCenter | Out-Null
        }
    }
    Add-TextBox $slide "validation acc1" 575 190 180 20 13 $C.Ink $true | Out-Null
    $labels = @("ANN","1 blk","2 blks","4 blks")
    $vals = @(94.1,74.3,67.6,51.5)
    $colors = @($C.Green,$C.Amber,$C.Amber,$C.Red)
    for ($i=0; $i -lt $vals.Count; $i++) {
        $barH = [int]($vals[$i] * 2.2)
        $x = 575 + $i*80
        $bar = $slide.Shapes.AddShape($msoShapeRectangle, $x, 410 - $barH, 42, $barH)
        $bar.Fill.ForeColor.RGB = $colors[$i]; $bar.Line.Visible = $msoFalse
        Add-TextBox $slide ([string]::Format("{0:N1}", $vals[$i])) ($x - 4) (388 - $barH) 55 16 8.5 $C.Ink $true $ppAlignCenter | Out-Null
        Add-TextBox $slide $labels[$i] ($x - 8) 418 60 14 8 $C.Muted $false $ppAlignCenter | Out-Null
    }
    Add-Card $slide 45 420 870 75 "结论" "失败路线说明 VideoMamba 的 residual、LayerNorm 与 Mamba 动态不适合简单阈值校准式转换，后续必须转向「加载 ANN 参数 + 插入脉冲层 + 再训练恢复」。" $C.Red
    Add-Footer $slide
}

function Build-TrainableRoute($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "可训练路线"
    Add-TextBox $slide "当前主线：基于 ANN 参数的可训练 SNN" 45 88 760 34 24 $C.Ink $true | Out-Null
    $steps = @(
        @("1","加载 ANN","clean VideoMamba`nbest.pth"),
        @("2","插入 LIF","post-block spike`nMamba 不动"),
        @("3","蒸馏训练","clean ANN teacher`n约束 logits"),
        @("4","分阶段扩展","0..2 → 0..5`n→ 0..11 → 0..23"),
        @("5","验证输出","24 层 active LIF`n均为 {0,1}")
    )
    for ($i=0; $i -lt $steps.Count; $i++) {
        $x = 45 + $i*175; $y = 205
        $box = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, 145, 105)
        $box.Fill.ForeColor.RGB = $C.Paper; $box.Line.ForeColor.RGB = $C.Line
        Add-Pill $slide $steps[$i][0] ($x + 12) ($y + 12) 30 24 $C.Cyan 10 | Out-Null
        Add-TextBox $slide $steps[$i][1] ($x + 50) ($y + 17) 85 18 10.5 $C.Ink $true | Out-Null
        Add-TextBox $slide $steps[$i][2] ($x + 12) ($y + 55) 120 38 9 $C.Muted $false $ppAlignCenter | Out-Null
    }
    Add-Card $slide 45 385 870 70 "时间步设置" "当前 SNN_TIMESTEPS=4：同一视频输入重复运行 4 次完整 forward，LIF 内部保留膜电位状态，最终平均 4 次 logits。" $C.Violet
    Add-Footer $slide
}

function Build-LayerCompare($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "脉冲层对比"
    Add-TextBox $slide "两种脉冲层尝试：精度友好 vs 部署友好" 45 88 760 34 24 $C.Ink $true | Out-Null
    Add-Card $slide 45 185 400 170 "TrainableSpike3dSeq" "输出：{-θ, 0, +θ}`n阈值：per-channel threshold`n结果：no-train 77.94；训练 best 96.32`n判断：精度高，但 signed 输出不利于重参数化。" $C.Amber
    Add-Card $slide 515 185 400 170 "SpikingJelly MultiStepLIFNode" "输出：{0, 1}`n神经元：tau=2.0, detach_reset=True`n结果：initial 25.74；训练 best 83.09`n判断：更符合真正脉冲传输和部署方向。" $C.Cyan
    Add-Card $slide 45 400 870 65 "当前选择" "继续以 unsigned LIF 作为主线。它更难训练，但 spike_stats 已确认 24 个 active LIF 层全部输出 {0,1}，更接近「数据以脉冲形式传输」的目标。" $C.Green
    Add-Footer $slide
}

function Build-Architecture($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "模型架构"
    Add-TextBox $slide "VideoMamba Unsigned LIF SNN：Mamba block 保持不动，block 输出脉冲化" 45 88 860 34 22 $C.Ink $true | Out-Null
    $items = @(
        @("Input video","B×3×16×224×224",$C.Blue),
        @("PatchEmbed","Conv3D`n384×16×14×14",$C.Cyan),
        @("Tokens + PE","3137×384`nCLS + pos",$C.Violet),
        @("24× blocks","VideoMamba Block_i`n→ LIF_i {0,1}",$C.Green),
        @("Final norm","residual + LN",$C.Amber),
        @("Head","mean pool`nLinear 12",$C.Red)
    )
    for ($i=0; $i -lt $items.Count; $i++) {
        $x = 45 + $i*145
        $w = $(if ($i -eq 3) { 135 } else { 118 })
        $box = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, 210, $w, 95)
        $box.Fill.ForeColor.RGB = $C.Paper; $box.Line.ForeColor.RGB = $items[$i][2]; $box.Line.Weight = 1.2
        Add-TextBox $slide $items[$i][0] ($x + 8) 224 ($w - 16) 16 9.5 $items[$i][2] $true $ppAlignCenter | Out-Null
        Add-TextBox $slide $items[$i][1] ($x + 8) 252 ($w - 16) 36 8.5 $C.Muted $false $ppAlignCenter | Out-Null
    }
    Add-Card $slide 45 385 260 70 "插入位置" "每个 Mamba block 后插入 LIF；patch_embed 不做 spike。" $C.Blue
    Add-Card $slide 350 385 260 70 "脉冲输出" "full 24-block 模型中 24 个 active LIF 层均输出 {0,1}。" $C.Cyan
    Add-Card $slide 655 385 260 70 "训练方式" "T=4，双 view，clean ANN teacher distillation。" $C.Violet
    Add-Footer $slide
}

function Build-LatestResults($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "最新结果"
    Add-TextBox $slide "Unsigned LIF 分阶段训练：脉冲化程度提高后仍能恢复到可接受精度" 45 88 850 34 21 $C.Ink $true | Out-Null
    $stages = @(
        @("ANN","94.12","baseline",$C.Green),
        @("0..2 LIF","91.18","best",$C.Cyan),
        @("0..5 LIF","88.97","best",$C.Cyan),
        @("0..11 LIF","78.68","best",$C.Amber),
        @("0..23 LIF","83.09","best",$C.Blue)
    )
    for ($i=0; $i -lt $stages.Count; $i++) {
        $x = 45 + $i*175
        $box = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, 180, 145, 78)
        $box.Fill.ForeColor.RGB = $C.Paper; $box.Line.ForeColor.RGB = $C.Line
        Add-TextBox $slide $stages[$i][0] ($x + 10) 194 120 15 9.5 $C.Muted $true $ppAlignCenter | Out-Null
        Add-TextBox $slide $stages[$i][1] ($x + 10) 216 120 25 18 $stages[$i][3] $true $ppAlignCenter | Out-Null
        Add-TextBox $slide $stages[$i][2] ($x + 10) 242 120 12 7.5 $C.Muted $false $ppAlignCenter | Out-Null
    }
    $rows = @(
        @("0..2","initial 52.21 → best 91.18 → latest 91.18"),
        @("0..5","initial 59.56 → best 88.97 → latest 88.97"),
        @("0..11","initial 38.97 → best 78.68 → latest 78.68"),
        @("0..23","initial 25.74 → best 83.09 → latest 81.62")
    )
    for ($i=0; $i -lt $rows.Count; $i++) {
        $y = 305 + $i*28
        Add-Pill $slide $rows[$i][0] 45 $y 70 18 $C.Dark2 7.5 | Out-Null
        Add-TextBox $slide $rows[$i][1] 130 ($y + 2) 600 14 8.5 $C.Muted $false $ppAlignLeft | Out-Null
    }
    Add-Card $slide 45 420 870 75 "当前判断" "full 24-block 一次性初始精度只有 25.74%，但训练 3 个 epoch 已恢复到 83.09%，说明路线可行；后续重点是延长训练、降低学习率微调，并尝试更细粒度插入策略。" $C.Green
    Add-Footer $slide
}

function Build-Contribution($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "阶段贡献"
    Add-TextBox $slide "本阶段工作的价值：把「能不能做」推进成「该怎么做」" 45 88 820 34 23 $C.Ink $true | Out-Null
    Add-Card $slide 45 180 400 110 "1｜清理并固定实验工程" "精简项目结构、同步 GitHub、保留训练日志，形成可复现实验流程。" $C.Blue
    Add-Card $slide 515 180 400 110 "2｜验证直接转换不可行" "ANN2SNN 路线被定量否定，避免继续在错误方向上消耗时间。" $C.Red
    Add-Card $slide 45 330 400 110 "3｜比较两类脉冲层" "确认 signed spike 精度高但部署意义弱，unsigned LIF 更符合真正脉冲化目标。" $C.Amber
    Add-Card $slide 515 330 400 110 "4｜建立可训练 SNN 主线" "基于 ANN 预训练、teacher distillation 和分阶段扩展，full 24-block 已达到 83.09%。" $C.Green
    Add-Footer $slide
}

function Build-NextSteps($pres) {
    $slide = Add-BlankSlide $pres $C.Paper
    Add-SectionLabel $slide "下一步计划"
    Add-TextBox $slide "后续优先把 full 24-block LIF 从「可行」推到「稳定可用」" 45 88 830 34 23 $C.Ink $true | Out-Null
    Add-Card $slide 45 200 260 175 "短期：继续恢复精度" "以 24-block best checkpoint 为起点，降低学习率继续训练；观察 val 是否能稳定到 85+ 或 88+。" $C.Green
    Add-Card $slide 350 200 260 175 "中期：细化插入策略" "保持 Mamba block 不动，尝试 post-block 归一化、局部分组和阈值初始化，减少分布偏移。" $C.Cyan
    Add-Card $slide 655 200 260 175 "长期：部署可解释性" "持续记录 spike_stats：非零率、二值输出、层间分布；再评估能耗、稀疏性与硬件友好性。" $C.Violet
    Add-Footer $slide
}

function Build-Closing($pres) {
    $slide = Add-BlankSlide $pres $C.Dark
    Add-TextBox $slide "阶段结论" 50 70 180 32 22 (RGBColor 148 163 184) $true | Out-Null
    Add-TextBox $slide "本月的核心进展不是一次性完成 SNN，" 50 155 760 44 30 $C.Paper $true | Out-Null
    Add-TextBox $slide "而是筛掉不适配路线，建立可继续训练的脉冲化主线。" 50 205 800 44 30 $C.Paper $true | Out-Null
    Add-TextBox $slide "VideoMamba ANN baseline：94.12% validation acc1`n直接 ANN2SNN：插入 4 个 block 后降到 51.47%，路线暂不作为主线`nfull 24-block unsigned LIF：best 83.09%，已确认 24 层输出 {0,1}" 50 315 780 90 15 $C.Slate $false | Out-Null
    Add-Pill $slide "下一阶段目标：提升 full 24-block LIF 精度稳定性，并探索更细粒度脉冲化设计" 50 430 720 30 $C.Cyan 12 | Out-Null
    Add-Footer $slide "Monthly Research 2026.05｜End"
}

$powerpoint = New-Object -ComObject PowerPoint.Application
$pres = $powerpoint.Presentations.Add()

try {
    $pres.PageSetup.SlideWidth = 960
    $pres.PageSetup.SlideHeight = 540

    Build-Cover $pres
    Build-Agenda $pres
    Build-PartOne $pres

    if (Test-Path -LiteralPath $source) {
        $pres.Slides.InsertFromFile($source, 3) | Out-Null
    } else {
        throw "Missing source PPT: $source"
    }

    Build-PartTwo $pres
    Build-ProjectFrame $pres
    Build-ConversionFail $pres
    Build-TrainableRoute $pres
    Build-LayerCompare $pres
    Build-Architecture $pres
    Build-LatestResults $pres
    Build-Contribution $pres
    Build-NextSteps $pres
    Build-Closing $pres

    if (Test-Path -LiteralPath $out) {
        Remove-Item -LiteralPath $out -Force
    }
    $pres.SaveAs($out)

    for ($i = 1; $i -le $pres.Slides.Count; $i++) {
        $path = Join-Path $previewDir ("preview_{0:D2}.png" -f $i)
        $pres.Slides.Item($i).Export($path, "PNG", 1280, 720)
    }

    Write-Output ("saved={0}" -f $out)
    Write-Output ("slides={0}" -f $pres.Slides.Count)
}
finally {
    $pres.Close()
    $powerpoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint) | Out-Null
}
