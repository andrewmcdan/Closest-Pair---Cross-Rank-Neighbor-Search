[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Points = "10000,100000,1000000,10000000",

    [Parameter(Mandatory = $false)]
    [string]$Threads = "1,6,12",

    [Parameter(Mandatory = $false)]
    [double]$Distribution = 10.0,

    [Parameter(Mandatory = $false)]
    [int]$Repeats = 1,

    [Parameter(Mandatory = $false)]
    [string]$ExecutablePath = ".\build\Release\closest_pair.exe",

    [Parameter(Mandatory = $false)]
    [string]$OutputCsv = ".\benchmark_results.csv",

    [Parameter(Mandatory = $false)]
    [switch]$RunDivideConquer,

    [Parameter(Mandatory = $false)]
    [switch]$NoBruteForce,

    [Parameter(Mandatory = $false)]
    [switch]$IncludeRawOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-PositiveIntList {
    param(
        [string]$Value,
        [string]$Name
    )

    $items = @()
    foreach ($token in ($Value -split ",")) {
        $trimmed = $token.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }

        $parsed = 0
        if (-not [int]::TryParse($trimmed, [ref]$parsed) -or $parsed -le 0) {
            throw "Invalid $Name entry '$trimmed'. Expected positive integers separated by commas."
        }

        $items += $parsed
    }

    if ($items.Count -eq 0) {
        throw "No valid values found for '$Name'."
    }

    return ,$items
}

function Get-RegexValue {
    param(
        [string]$Text,
        [string]$Pattern
    )

    $match = [regex]::Match($Text, $Pattern)
    if ($match.Success) {
        return $match.Groups[1].Value
    }

    return $null
}

function To-NullableInt {
    param([string]$Value)
    if ($null -eq $Value) { return $null }
    $trimmed = $Value.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) { return $null }

    $parsed = 0
    if ([int]::TryParse($trimmed, [ref]$parsed)) {
        return $parsed
    }

    return $null
}

function To-NullableDouble {
    param([string]$Value)
    if ($null -eq $Value) { return $null }
    $trimmed = $Value.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) { return $null }

    $parsed = 0.0
    if ([double]::TryParse(
            $trimmed,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$parsed
        )) {
        return $parsed
    }

    if ([double]::TryParse(
            $trimmed,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::CurrentCulture,
            [ref]$parsed
        )) {
        return $parsed
    }

    return $null
}

function To-NullableBool {
    param([string]$Value)
    if ($null -eq $Value) { return $null }
    if ($Value -eq "true") { return $true }
    if ($Value -eq "false") { return $false }
    return $null
}

if ($Distribution -lt 0.0 -or $Distribution -gt 10.0) {
    throw "Distribution must be in [0, 10]."
}

if ($Repeats -le 0) {
    throw "Repeats must be a positive integer."
}

$pointValues = Parse-PositiveIntList -Value $Points -Name "points"
$threadValues = Parse-PositiveIntList -Value $Threads -Name "threads"

if (-not (Test-Path $ExecutablePath)) {
    throw "Executable not found at '$ExecutablePath'. Build first (e.g. cmake --build build --config Release)."
}

$exe = (Resolve-Path $ExecutablePath).Path
$results = New-Object System.Collections.Generic.List[object]

$totalRuns = $pointValues.Count * $threadValues.Count * $Repeats
$runIndex = 0

Write-Host "Executable: $exe"
Write-Host "Points: $($pointValues -join ', ')"
Write-Host "Threads: $($threadValues -join ', ')"
Write-Host "Distribution: $Distribution"
Write-Host "Repeats: $Repeats"
Write-Host "RunDivideConquer: $([bool]$RunDivideConquer)"
Write-Host "NoBruteForce: $([bool]$NoBruteForce)"
Write-Host ""

foreach ($p in $pointValues) {
    foreach ($t in $threadValues) {
        for ($r = 1; $r -le $Repeats; ++$r) {
            $runIndex += 1

            $args = @(
                "--points", "$p",
                "--threads", "$t",
                "--distribution", "$Distribution"
            )

            if ($RunDivideConquer) {
                $args += "--run-divide-conquer"
            }

            if ($NoBruteForce) {
                $args += "--no-bruteforce"
            }

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $outputLines = & $exe @args 2>&1
            $exitCode = $LASTEXITCODE
            $sw.Stop()

            $text = ($outputLines | ForEach-Object { $_.ToString() }) -join "`n"

            $reportedPoints = To-NullableInt (Get-RegexValue -Text $text -Pattern "Point count:\s*(\d+)")
            $reportedThreads = To-NullableInt (Get-RegexValue -Text $text -Pattern "Thread count:\s*(\d+)")
            $reportedDistribution = To-NullableDouble (Get-RegexValue -Text $text -Pattern "Distribution modifier:\s*([+-]?\d+(?:\.\d+)?)")

            $crnsSerialMs = To-NullableDouble (Get-RegexValue -Text $text -Pattern "CRNS serial time \(ms\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
            $crnsParallelMs = To-NullableDouble (Get-RegexValue -Text $text -Pattern "CRNS parallel time \(ms\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

            $divideSerialMs = To-NullableDouble (Get-RegexValue -Text $text -Pattern "Divide-and-conquer serial time \(ms\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
            $divideParallelMs = To-NullableDouble (Get-RegexValue -Text $text -Pattern "Divide-and-conquer parallel time \(ms\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
            $bruteMs = To-NullableDouble (Get-RegexValue -Text $text -Pattern "Brute force time \(ms\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

            $crnsSerialMatchesBrute = To-NullableBool (Get-RegexValue -Text $text -Pattern "CRNS serial matches brute force:\s*(true|false)")
            $crnsParallelMatchesBrute = To-NullableBool (Get-RegexValue -Text $text -Pattern "CRNS parallel matches brute force:\s*(true|false)")

            $divideSerialMatchesBrute = To-NullableBool (Get-RegexValue -Text $text -Pattern "Divide-and-conquer serial matches brute force:\s*(true|false)")
            $divideParallelMatchesBrute = To-NullableBool (Get-RegexValue -Text $text -Pattern "Divide-and-conquer parallel matches brute force:\s*(true|false)")

            $crnsSerialMatchesDivideSerial = To-NullableBool (Get-RegexValue -Text $text -Pattern "CRNS serial matches divide-and-conquer serial:\s*(true|false)")
            $crnsParallelMatchesDivideParallel = To-NullableBool (Get-RegexValue -Text $text -Pattern "CRNS parallel matches divide-and-conquer parallel:\s*(true|false)")

            $bruteDisabled = ((Get-RegexValue -Text $text -Pattern "Brute force check:\s*(disabled)") -eq "disabled")
            $comparisonReductionCrnsSerial = To-NullableDouble (Get-RegexValue -Text $text -Pattern "Comparison reduction factor \(brute / CRNS serial\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
            $comparisonReductionCrnsParallel = To-NullableDouble (Get-RegexValue -Text $text -Pattern "Comparison reduction factor \(brute / CRNS parallel\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

            $row = [pscustomobject]@{
                run_index                              = $runIndex
                repeat_index                           = $r
                requested_points                       = $p
                requested_threads                      = $t
                requested_distribution                 = $Distribution
                run_divide_conquer                     = [bool]$RunDivideConquer
                no_bruteforce                          = [bool]$NoBruteForce
                exit_code                              = $exitCode
                wall_time_ms                           = [math]::Round($sw.Elapsed.TotalMilliseconds, 6)
                reported_points                        = $reportedPoints
                reported_threads                       = $reportedThreads
                reported_distribution                  = $reportedDistribution
                crns_serial_ms                         = $crnsSerialMs
                crns_parallel_ms                       = $crnsParallelMs
                divide_conquer_serial_ms               = $divideSerialMs
                divide_conquer_parallel_ms             = $divideParallelMs
                brute_force_ms                         = $bruteMs
                comparison_reduction_brute_over_crns_serial = $comparisonReductionCrnsSerial
                comparison_reduction_brute_over_crns_parallel = $comparisonReductionCrnsParallel
                crns_serial_matches_bruteforce         = $crnsSerialMatchesBrute
                crns_parallel_matches_bruteforce       = $crnsParallelMatchesBrute
                divide_serial_matches_bruteforce       = $divideSerialMatchesBrute
                divide_parallel_matches_bruteforce     = $divideParallelMatchesBrute
                crns_serial_matches_divide_serial      = $crnsSerialMatchesDivideSerial
                crns_parallel_matches_divide_parallel  = $crnsParallelMatchesDivideParallel
                brute_force_disabled                   = $bruteDisabled
                raw_output                             = $(if ($IncludeRawOutput) { $text } else { $null })
            }

            $results.Add($row)

            Write-Host ("[{0}/{1}] points={2} threads={3} repeat={4} crns_serial_ms={5} crns_parallel_ms={6}" -f `
                $runIndex, $totalRuns, $p, $t, $r, `
                $(if ($null -ne $crnsSerialMs) { [math]::Round($crnsSerialMs, 6) } else { "n/a" }), `
                $(if ($null -ne $crnsParallelMs) { [math]::Round($crnsParallelMs, 6) } else { "n/a" }))
        }
    }
}

$outputPath = [System.IO.Path]::GetFullPath($OutputCsv)
$outputDir = [System.IO.Path]::GetDirectoryName($outputPath)
if (-not [string]::IsNullOrWhiteSpace($outputDir) -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$results | Export-Csv -Path $outputPath -NoTypeInformation -Encoding UTF8
Write-Host ""
Write-Host "Wrote $($results.Count) rows to: $outputPath"
