param(
    [int]$MaxJobs = [Environment]::ProcessorCount,
    [int[]]$RngSeeds = 1..50,
    [int]$TotalSteps = 1000
)

$TOPDIR = (Resolve-Path $PSScriptRoot).Path
$env:PYTHONHASHSEED = '0'

$chainOut  = Join-Path $TOPDIR 'chain_outputs'
$chainLogs = Join-Path $TOPDIR 'chain_logs'
New-Item -ItemType Directory -Force -Path $chainOut, $chainLogs | Out-Null

# Resolve uv once so jobs don't depend on profile PATH
$uvExe = (Get-Command uv -ErrorAction Stop).Source

$jobs = @()

foreach ($seed in $RngSeeds)
{

    # throttle
    while (($jobs | Where-Object State -eq 'Running').Count -ge $MaxJobs)
    {
        Start-Sleep -Milliseconds 200
        $done = $jobs | Where-Object State -in 'Completed','Failed','Stopped'
        if ($done)
        {
            Receive-Job -Job $done -Keep | Out-Null
            $jobs = $jobs | Where-Object State -in 'Running','NotStarted'
        }
    }

    $outFile = Join-Path $chainOut ("gerrymandria_chain_{0}_steps_seed{1}.jsonl" -f $TotalSteps, $seed)
    $logFile = Join-Path $chainLogs ("log_parallel_rng_seed_{0}.log" -f $seed)

    $job = Start-Job -Name "seed$seed" `
        -ArgumentList $TOPDIR, $TotalSteps, $seed, $outFile, $logFile, $uvExe `
        -ScriptBlock {
        param($topdir, $nsteps, $seed, $outFile, $logFile, $uvExe)

        Set-StrictMode -Version Latest
        $ErrorActionPreference = 'Stop'
        $env:PYTHONHASHSEED = '0'

        # Cross-platform paths
        $exampleCli = Join-Path $topdir (Join-Path 'pipeline_scripts' 'example_cli.py')
        $graphPath  = Join-Path $topdir (Join-Path 'JSON_dualgraphs' 'gerrymandria.json')

        # Build args as an array
        $arguments = @(
            'run', '--project', $topdir, $exampleCli,
            '--graph-path', $graphPath,
            '--output-path', $outFile,
            '--starting-plan', 'district',
            '--pop-col', 'TOTPOP',
            '--rng-seed', $seed,
            '--population-tolerance', '0.01',
            '--total-steps', $nsteps,
            '--writeas', 'jsonl'
        )

        try
        {
            & $uvExe @arguments *> $logFile
        } catch
        {
            $_ | Out-String | Add-Content $logFile
            throw
        }
    }

    $jobs += $job
}

Write-Progress -Activity "Running jobs" -Status "Waiting for completion..."
Wait-Job -Job $jobs
Receive-Job -Job $jobs -Keep | Out-Null
Write-Progress -Activity "Running jobs" -Completed
