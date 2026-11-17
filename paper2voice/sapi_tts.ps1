param(
    [Parameter(Mandatory = $true)][string]$TextPath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [string]$Voice = "Microsoft Huihui Desktop",
    [int]$Rate = -1
)

$text = [System.IO.File]::ReadAllText($TextPath, [System.Text.Encoding]::UTF8)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $selected = $synth.GetInstalledVoices() |
        Where-Object { $_.VoiceInfo.Name -eq $Voice -and $_.Enabled } |
        Select-Object -First 1
    if ($selected) {
        $synth.SelectVoice($selected.VoiceInfo.Name)
    } else {
        $chinese = $synth.GetInstalledVoices() |
            Where-Object { $_.VoiceInfo.Culture.Name -eq "zh-CN" -and $_.Enabled } |
            Select-Object -First 1
        if ($chinese) {
            $synth.SelectVoice($chinese.VoiceInfo.Name)
        }
    }
    $synth.Rate = $Rate
    $synth.SetOutputToWaveFile($OutputPath)
    $synth.Speak($text)
} finally {
    $synth.Dispose()
}
