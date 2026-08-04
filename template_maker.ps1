# ---------------------------------------------------------------------------
# THIS FILE IS GENERATED from installer_src/skeleton.ps1 and template_project/.
# Edit those sources and run 'python3 generate_installers.py' instead of
# editing this script directly.
# ---------------------------------------------------------------------------

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'


# =====================================
# ========  UTILITY FUNCTIONS  ========
# =====================================

function Write-Info($msg)
{ Write-Host "[*] $msg" -ForegroundColor Cyan 
}
function Write-OK($msg)
{ Write-Host "[OK] $msg" -ForegroundColor Green 
}
function Write-Warn($msg)
{ Write-Host "[!] $msg" -ForegroundColor Yellow 
}
function Write-Err($msg)
{ Write-Host "[X] $msg" -ForegroundColor Red 
}

function Test-Command
{
    param([Parameter(Mandatory)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-WithRetry
{
    param(
        [Parameter(Mandatory)] [scriptblock] $Action,
        [int] $MaxAttempts = 5,
        [int] $InitialDelaySeconds = 2
    )

    $attempt = 1
    $delay = $InitialDelaySeconds

    while ($true)
    {
        try
        {
            return & $Action
        } catch
        {
            if ($attempt -ge $MaxAttempts)
            {
                throw  # rethrow last error after max attempts
            }

            Write-Warn "Attempt $attempt failed: $($_.Exception.Message)"
            Write-Info "Retrying in $delay seconds..."
            Start-Sleep -Seconds $delay

            $attempt++
            # simple backoff (cap it a bit)
            $delay = [Math]::Min($delay * 2, 30)
        }
    }
}

# ====================================================
# ========  SOFTWARE CHECKERS AND INSTALLERS  ========
# ====================================================

function Confirm-Uv
{
    if (Test-Command -Name 'uv')
    { return 
    }
    $choice = Read-Host "uv not found. Install it now? (y/[n])"
    if ($choice -notin @('y','Y'))
    {
        Write-Err "uv is required to run this script. Exiting."
        exit 1
    }
    Write-Info "Installing uv..."
    try
    {
        # Recommended installer
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # Common install path
        $uvBin = Join-Path $HOME ".local\bin"
        if (Test-Path $uvBin)
        { $env:Path = "$uvBin;$env:Path" 
        }
    } catch
    {
        Write-Err "uv installation failed. See https://docs.astral.sh/uv/getting-started/installation/"
        throw
    }
    if (-not (Test-Command -Name 'uv'))
    {
        Write-Err "uv still not found on PATH after install."
        throw "uv not found"
    }
    Write-OK "uv installed."
}

function Confirm-BuildTools
{
    param(
        [bool]$InstallIfMissing = $true,
        [bool]$RequireWinSDK    = $true
    )

    function Test-CppToolchain
    {
        $hasLink = [bool](Get-Command link.exe -ErrorAction SilentlyContinue)
        $hasCl   = [bool](Get-Command cl.exe   -ErrorAction SilentlyContinue)


        if (-not $RequireWinSDK)
        {
            $sdkOk = $true
        } else
        {
            $sdkOk = $false
        }

        if ($RequireWinSDK)
        {
            $candidates = @()

            # 1) Registry
            try
            {
                $roots = Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots' -ErrorAction Stop
                if ($roots -and $roots.PSObject.Properties.Name -contains 'KitsRoot10')
                {
                    $candidates += $roots.KitsRoot10
                }
            } catch
            { 
            }

            # 2) Env var
            if ($env:WindowsSdkDir)
            { $candidates += $env:WindowsSdkDir 
            }

            # 3) Common locations
            $candidates += @(
                'C:\Program Files (x86)\Windows Kits\10\',
                'C:\Program Files\Windows Kits\10\'
            )

            foreach ($root in $candidates | Where-Object { $_ -and (Test-Path $_) })
            {
                if (Test-Path (Join-Path $root 'Lib'))
                { $sdkOk = $true; break 
                }
            }
        }

        [pscustomobject]@{
            Link = $hasLink
            Cl   = $hasCl
            Sdk  = $sdkOk
        }
    }

    function Update-MsvcPath
    {
        $pf86 = ${env:ProgramFiles(x86)}
        if (-not $pf86)
        { return 
        }
        $vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path $vswhere))
        { return 
        }

        $vsPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if (-not $vsPath)
        { return 
        }

        $toolRoot = Join-Path $vsPath 'VC\Tools\MSVC'
        if (-not (Test-Path $toolRoot))
        { return 
        }

        $latest = Get-ChildItem $toolRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $latest)
        { return 
        }

        $binCandidates = @(
            Join-Path $latest.FullName 'bin\Hostx64\x64'
            Join-Path $latest.FullName 'bin\Hostx86\x64'
            Join-Path $latest.FullName 'bin\Hostx64\x86'
            Join-Path $latest.FullName 'bin\Hostx86\x86'
        ) | Where-Object { Test-Path $_ }

        foreach ($bin in $binCandidates)
        {
            $escaped = [regex]::Escape($bin)
            if ($env:Path -notmatch "(^|;)$escaped(;|$)")
            {
                $env:Path = "$bin;$env:Path"
            }
        }
    }

    Write-Info "Checking MSVC toolchain (cl/link) and Windows SDK..."
    Update-MsvcPath
    $state = Test-CppToolchain
    if ($state.Link -and $state.Cl -and $state.Sdk)
    {
        Write-OK "MSVC & Windows SDK detected."
        if (Get-Command rustup -ErrorAction SilentlyContinue)
        {
            try
            { & rustup default stable-x86_64-pc-windows-msvc | Out-Null 
            } catch
            {
            }
        }
        return $true
    }

    if (-not $InstallIfMissing)
    {
        Write-Err "MSVC build tools or Windows SDK missing."
        throw "Build tools not present."
    }

    if (-not (Test-Command -Name 'winget'))
    {
        Write-Err "winget not found. Install Build Tools manually via Visual Studio Installer."
        throw "winget missing"
    }

    Write-Warn "Installing Visual Studio 2022 Build Tools (C++ workload + SDK)... (this can take a while)"
    $override = @(
        '--quiet','--wait','--norestart',
        '--add','Microsoft.VisualStudio.Workload.VCTools',
        '--includeRecommended'
    ) -join ' '

    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "$override"

    Update-MsvcPath
    $state = Test-CppToolchain
    if (-not ($state.Link -and $state.Cl -and $state.Sdk))
    {
        Write-Err "MSVC/SDK still not detected after install."
        Write-Info "Open 'Visual Studio Installer' -> Modify 'Build Tools' -> ensure 'C++ build tools' + a Windows 10/11 SDK are selected."
        throw "Build tools not detected"
    }

    Write-OK "MSVC build tools ready."
    if (Test-Command -Name 'rustup')
    {
        try
        {
            & rustup default stable-x86_64-pc-windows-msvc | Out-Null
            & rustup component add rustfmt clippy | Out-Null
        } catch
        {
        }
    }
    return $true
}

function Confirm-Cargo
{
    if (Test-Command -Name 'cargo')
    {
        $cargoBin = Join-Path $HOME ".cargo\bin"
        if (Test-Path $cargoBin)
        { $env:Path = "$cargoBin;$env:Path" 
        }
        return
    }
    $choice = Read-Host "Cargo not found. Install Rust/Cargo via rustup now? (y/[n])"
    if ($choice -notin @('y','Y'))
    {
        Write-Err "Cargo is required for FRCW/BEN path. Exiting."
        exit 1
    }
    Write-Info "Installing Rust/Cargo (rustup)..."
    try
    {
        if (Test-Command -Name 'winget')
        {
            winget install Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements
        } else
        {
            $tmp = Join-Path $env:TEMP "rustup-init.exe"
            Invoke-WebRequest "https://win.rustup.rs/x86_64" -OutFile $tmp
            & $tmp -y
        }
        $cargoBin = Join-Path $HOME ".cargo\bin"
        if (Test-Path $cargoBin)
        { $env:Path = "$cargoBin;$env:Path" 
        }
    } catch
    {
        Write-Err "Rust/Cargo installation failed. Install from https://www.rust-lang.org/tools/install and re-run."
        throw
    }
    if (-not (Test-Command -Name 'cargo'))
    {
        Write-Err "cargo still not found on PATH."
        throw "cargo not found"
    }
    Write-OK "Rust and Cargo installed."
}

# ==================================================
# ========  EMBEDDED PROJECT FILES  ================
# ==================================================

# ====  GENERATED PAYLOADS (from template_project/) -- DO NOT EDIT BY HAND  ====
# ====  regenerate with: python3 generate_installers.py                     ====

$PayloadDirectories = @(
    'JSON_dualgraphs'
    'chain_logs'
    'chain_outputs'
    'data'
    'dev_files'
    'figures'
    'notebooks'
    'pipeline_scripts'
    'pipeline_scripts/metrics'
    'stats'
)

$Payloads = [ordered]@{
'.env' = @'
PYTHONHASHSEED=0
'@
'.gitignore' = @'
.venv/
__pycache__/
*.py[cod]
uv.lock
dev_files/*
chain_logs/*
chain_outputs/*
figures/*
stats/*

!.gitkeep
!dev_files/.gitkeep
!chain_logs/.gitkeep
!chain_outputs/.gitkeep
!figures/.gitkeep
!stats/.gitkeep
'@
'.python-version' = @'
3.11
'@
'JSON_dualgraphs/gerrymandria.json' = @'
{
    "directed": false,
    "multigraph": false,
    "graph": [],
    "nodes": [
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 0,
            "county": "1",
            "district": "1",
            "precinct": 0,
            "muni": "1",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 0
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 1,
            "county": "1",
            "district": "1",
            "precinct": 1,
            "muni": "1",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 1
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 2,
            "county": "1",
            "district": "1",
            "precinct": 2,
            "muni": "5",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 2
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 3,
            "county": "1",
            "district": "1",
            "precinct": 3,
            "muni": "5",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 3
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 4,
            "county": "3",
            "district": "1",
            "precinct": 4,
            "muni": "9",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 4
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 5,
            "county": "3",
            "district": "1",
            "precinct": 5,
            "muni": "9",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 5
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 6,
            "county": "3",
            "district": "1",
            "precinct": 6,
            "muni": "13",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 6
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 7,
            "county": "3",
            "district": "1",
            "precinct": 7,
            "muni": "13",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 7
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 0,
            "county": "1",
            "district": "2",
            "precinct": 8,
            "muni": "1",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 8
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 1,
            "county": "1",
            "district": "2",
            "precinct": 9,
            "muni": "1",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 9
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 2,
            "county": "1",
            "district": "2",
            "precinct": 10,
            "muni": "5",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 10
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 3,
            "county": "1",
            "district": "2",
            "precinct": 11,
            "muni": "5",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 11
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 4,
            "county": "3",
            "district": "2",
            "precinct": 12,
            "muni": "9",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 12
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 5,
            "county": "3",
            "district": "2",
            "precinct": 13,
            "muni": "9",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 13
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 6,
            "county": "3",
            "district": "2",
            "precinct": 14,
            "muni": "13",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 14
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 7,
            "county": "3",
            "district": "2",
            "precinct": 15,
            "muni": "13",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 15
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 0,
            "county": "1",
            "district": "3",
            "precinct": 16,
            "muni": "2",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 16
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 1,
            "county": "1",
            "district": "3",
            "precinct": 17,
            "muni": "2",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 17
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 2,
            "county": "1",
            "district": "3",
            "precinct": 18,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 18
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 3,
            "county": "1",
            "district": "3",
            "precinct": 19,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 19
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 4,
            "county": "3",
            "district": "3",
            "precinct": 20,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 20
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 5,
            "county": "3",
            "district": "3",
            "precinct": 21,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 21
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 6,
            "county": "3",
            "district": "3",
            "precinct": 22,
            "muni": "14",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 22
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 7,
            "county": "3",
            "district": "3",
            "precinct": 23,
            "muni": "14",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 23
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 0,
            "county": "1",
            "district": "4",
            "precinct": 24,
            "muni": "2",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 24
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 1,
            "county": "1",
            "district": "4",
            "precinct": 25,
            "muni": "2",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 25
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 2,
            "county": "1",
            "district": "4",
            "precinct": 26,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 26
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 3,
            "county": "1",
            "district": "4",
            "precinct": 27,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 27
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 4,
            "county": "3",
            "district": "4",
            "precinct": 28,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 28
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 5,
            "county": "3",
            "district": "4",
            "precinct": 29,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 29
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 6,
            "county": "3",
            "district": "4",
            "precinct": 30,
            "muni": "14",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 30
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 7,
            "county": "3",
            "district": "4",
            "precinct": 31,
            "muni": "14",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 31
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 0,
            "county": "2",
            "district": "5",
            "precinct": 32,
            "muni": "3",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 32
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 1,
            "county": "2",
            "district": "5",
            "precinct": 33,
            "muni": "3",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 33
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 2,
            "county": "2",
            "district": "5",
            "precinct": 34,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 34
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 3,
            "county": "2",
            "district": "5",
            "precinct": 35,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 35
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 4,
            "county": "4",
            "district": "5",
            "precinct": 36,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 36
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 5,
            "county": "4",
            "district": "5",
            "precinct": 37,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 37
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 6,
            "county": "4",
            "district": "5",
            "precinct": 38,
            "muni": "15",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 38
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 7,
            "county": "4",
            "district": "5",
            "precinct": 39,
            "muni": "15",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 39
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 0,
            "county": "2",
            "district": "6",
            "precinct": 40,
            "muni": "3",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 40
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 1,
            "county": "2",
            "district": "6",
            "precinct": 41,
            "muni": "3",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 41
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 2,
            "county": "2",
            "district": "6",
            "precinct": 42,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 42
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 3,
            "county": "2",
            "district": "6",
            "precinct": 43,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 43
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 4,
            "county": "4",
            "district": "6",
            "precinct": 44,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 44
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 5,
            "county": "4",
            "district": "6",
            "precinct": 45,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 45
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 6,
            "county": "4",
            "district": "6",
            "precinct": 46,
            "muni": "15",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 46
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 7,
            "county": "4",
            "district": "6",
            "precinct": 47,
            "muni": "15",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 47
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 0,
            "county": "2",
            "district": "7",
            "precinct": 48,
            "muni": "4",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 48
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 1,
            "county": "2",
            "district": "7",
            "precinct": 49,
            "muni": "4",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 49
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 2,
            "county": "2",
            "district": "7",
            "precinct": 50,
            "muni": "8",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 50
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 3,
            "county": "2",
            "district": "7",
            "precinct": 51,
            "muni": "8",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 51
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 4,
            "county": "4",
            "district": "7",
            "precinct": 52,
            "muni": "12",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 52
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 5,
            "county": "4",
            "district": "7",
            "precinct": 53,
            "muni": "12",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 53
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 6,
            "county": "4",
            "district": "7",
            "precinct": 54,
            "muni": "16",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 54
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 7,
            "county": "4",
            "district": "7",
            "precinct": 55,
            "muni": "16",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 55
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 0,
            "county": "2",
            "district": "8",
            "precinct": 56,
            "muni": "4",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 56
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 1,
            "county": "2",
            "district": "8",
            "precinct": 57,
            "muni": "4",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 57
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 2,
            "county": "2",
            "district": "8",
            "precinct": 58,
            "muni": "8",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 58
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 3,
            "county": "2",
            "district": "8",
            "precinct": 59,
            "muni": "8",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 59
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 4,
            "county": "4",
            "district": "8",
            "precinct": 60,
            "muni": "12",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 60
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 5,
            "county": "4",
            "district": "8",
            "precinct": 61,
            "muni": "12",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 61
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 6,
            "county": "4",
            "district": "8",
            "precinct": 62,
            "muni": "16",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 62
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 7,
            "county": "4",
            "district": "8",
            "precinct": 63,
            "muni": "16",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 63
        }
    ],
    "adjacency": [
        [
            {
                "id": 8
            },
            {
                "id": 1
            }
        ],
        [
            {
                "id": 0
            },
            {
                "id": 9
            },
            {
                "id": 2
            }
        ],
        [
            {
                "id": 1
            },
            {
                "id": 10
            },
            {
                "id": 3
            }
        ],
        [
            {
                "id": 2
            },
            {
                "id": 11
            },
            {
                "id": 4
            }
        ],
        [
            {
                "id": 3
            },
            {
                "id": 12
            },
            {
                "id": 5
            }
        ],
        [
            {
                "id": 4
            },
            {
                "id": 13
            },
            {
                "id": 6
            }
        ],
        [
            {
                "id": 5
            },
            {
                "id": 14
            },
            {
                "id": 7
            }
        ],
        [
            {
                "id": 6
            },
            {
                "id": 15
            }
        ],
        [
            {
                "id": 0
            },
            {
                "id": 16
            },
            {
                "id": 9
            }
        ],
        [
            {
                "id": 1
            },
            {
                "id": 8
            },
            {
                "id": 17
            },
            {
                "id": 10
            }
        ],
        [
            {
                "id": 2
            },
            {
                "id": 9
            },
            {
                "id": 18
            },
            {
                "id": 11
            }
        ],
        [
            {
                "id": 3
            },
            {
                "id": 10
            },
            {
                "id": 19
            },
            {
                "id": 12
            }
        ],
        [
            {
                "id": 4
            },
            {
                "id": 11
            },
            {
                "id": 20
            },
            {
                "id": 13
            }
        ],
        [
            {
                "id": 5
            },
            {
                "id": 12
            },
            {
                "id": 21
            },
            {
                "id": 14
            }
        ],
        [
            {
                "id": 6
            },
            {
                "id": 13
            },
            {
                "id": 22
            },
            {
                "id": 15
            }
        ],
        [
            {
                "id": 7
            },
            {
                "id": 14
            },
            {
                "id": 23
            }
        ],
        [
            {
                "id": 8
            },
            {
                "id": 24
            },
            {
                "id": 17
            }
        ],
        [
            {
                "id": 9
            },
            {
                "id": 16
            },
            {
                "id": 25
            },
            {
                "id": 18
            }
        ],
        [
            {
                "id": 10
            },
            {
                "id": 17
            },
            {
                "id": 26
            },
            {
                "id": 19
            }
        ],
        [
            {
                "id": 11
            },
            {
                "id": 18
            },
            {
                "id": 27
            },
            {
                "id": 20
            }
        ],
        [
            {
                "id": 12
            },
            {
                "id": 19
            },
            {
                "id": 28
            },
            {
                "id": 21
            }
        ],
        [
            {
                "id": 13
            },
            {
                "id": 20
            },
            {
                "id": 29
            },
            {
                "id": 22
            }
        ],
        [
            {
                "id": 14
            },
            {
                "id": 21
            },
            {
                "id": 30
            },
            {
                "id": 23
            }
        ],
        [
            {
                "id": 15
            },
            {
                "id": 22
            },
            {
                "id": 31
            }
        ],
        [
            {
                "id": 16
            },
            {
                "id": 32
            },
            {
                "id": 25
            }
        ],
        [
            {
                "id": 17
            },
            {
                "id": 24
            },
            {
                "id": 33
            },
            {
                "id": 26
            }
        ],
        [
            {
                "id": 18
            },
            {
                "id": 25
            },
            {
                "id": 34
            },
            {
                "id": 27
            }
        ],
        [
            {
                "id": 19
            },
            {
                "id": 26
            },
            {
                "id": 35
            },
            {
                "id": 28
            }
        ],
        [
            {
                "id": 20
            },
            {
                "id": 27
            },
            {
                "id": 36
            },
            {
                "id": 29
            }
        ],
        [
            {
                "id": 21
            },
            {
                "id": 28
            },
            {
                "id": 37
            },
            {
                "id": 30
            }
        ],
        [
            {
                "id": 22
            },
            {
                "id": 29
            },
            {
                "id": 38
            },
            {
                "id": 31
            }
        ],
        [
            {
                "id": 23
            },
            {
                "id": 30
            },
            {
                "id": 39
            }
        ],
        [
            {
                "id": 24
            },
            {
                "id": 40
            },
            {
                "id": 33
            }
        ],
        [
            {
                "id": 25
            },
            {
                "id": 32
            },
            {
                "id": 41
            },
            {
                "id": 34
            }
        ],
        [
            {
                "id": 26
            },
            {
                "id": 33
            },
            {
                "id": 42
            },
            {
                "id": 35
            }
        ],
        [
            {
                "id": 27
            },
            {
                "id": 34
            },
            {
                "id": 43
            },
            {
                "id": 36
            }
        ],
        [
            {
                "id": 28
            },
            {
                "id": 35
            },
            {
                "id": 44
            },
            {
                "id": 37
            }
        ],
        [
            {
                "id": 29
            },
            {
                "id": 36
            },
            {
                "id": 45
            },
            {
                "id": 38
            }
        ],
        [
            {
                "id": 30
            },
            {
                "id": 37
            },
            {
                "id": 46
            },
            {
                "id": 39
            }
        ],
        [
            {
                "id": 31
            },
            {
                "id": 38
            },
            {
                "id": 47
            }
        ],
        [
            {
                "id": 32
            },
            {
                "id": 48
            },
            {
                "id": 41
            }
        ],
        [
            {
                "id": 33
            },
            {
                "id": 40
            },
            {
                "id": 49
            },
            {
                "id": 42
            }
        ],
        [
            {
                "id": 34
            },
            {
                "id": 41
            },
            {
                "id": 50
            },
            {
                "id": 43
            }
        ],
        [
            {
                "id": 35
            },
            {
                "id": 42
            },
            {
                "id": 51
            },
            {
                "id": 44
            }
        ],
        [
            {
                "id": 36
            },
            {
                "id": 43
            },
            {
                "id": 52
            },
            {
                "id": 45
            }
        ],
        [
            {
                "id": 37
            },
            {
                "id": 44
            },
            {
                "id": 53
            },
            {
                "id": 46
            }
        ],
        [
            {
                "id": 38
            },
            {
                "id": 45
            },
            {
                "id": 54
            },
            {
                "id": 47
            }
        ],
        [
            {
                "id": 39
            },
            {
                "id": 46
            },
            {
                "id": 55
            }
        ],
        [
            {
                "id": 40
            },
            {
                "id": 56
            },
            {
                "id": 49
            }
        ],
        [
            {
                "id": 41
            },
            {
                "id": 48
            },
            {
                "id": 57
            },
            {
                "id": 50
            }
        ],
        [
            {
                "id": 42
            },
            {
                "id": 49
            },
            {
                "id": 58
            },
            {
                "id": 51
            }
        ],
        [
            {
                "id": 43
            },
            {
                "id": 50
            },
            {
                "id": 59
            },
            {
                "id": 52
            }
        ],
        [
            {
                "id": 44
            },
            {
                "id": 51
            },
            {
                "id": 60
            },
            {
                "id": 53
            }
        ],
        [
            {
                "id": 45
            },
            {
                "id": 52
            },
            {
                "id": 61
            },
            {
                "id": 54
            }
        ],
        [
            {
                "id": 46
            },
            {
                "id": 53
            },
            {
                "id": 62
            },
            {
                "id": 55
            }
        ],
        [
            {
                "id": 47
            },
            {
                "id": 54
            },
            {
                "id": 63
            }
        ],
        [
            {
                "id": 48
            },
            {
                "id": 57
            }
        ],
        [
            {
                "id": 49
            },
            {
                "id": 56
            },
            {
                "id": 58
            }
        ],
        [
            {
                "id": 50
            },
            {
                "id": 57
            },
            {
                "id": 59
            }
        ],
        [
            {
                "id": 51
            },
            {
                "id": 58
            },
            {
                "id": 60
            }
        ],
        [
            {
                "id": 52
            },
            {
                "id": 59
            },
            {
                "id": 61
            }
        ],
        [
            {
                "id": 53
            },
            {
                "id": 60
            },
            {
                "id": 62
            }
        ],
        [
            {
                "id": 54
            },
            {
                "id": 61
            },
            {
                "id": 63
            }
        ],
        [
            {
                "id": 55
            },
            {
                "id": 62
            }
        ]
    ]
}
'@
'README.md' = @'
# Redistricting Project

This project contains example scripts for running redistricting chains, converting
ensemble files, and calculating common metrics.

Install the Python environment with:

```bash
uv sync
```

The Bash and PowerShell helper scripts are both kept here so the project can be used as
the source for the platform-specific Democracy Batsignal installers.
'@
'batch_example_python_cli_parallel.ps1' = @'
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
'@
'batch_example_python_cli_simple.ps1' = @'
param(
  [int[]]$RngSeeds = @(42,43,44),
  [int]$TotalSteps = 1000,
  [int[]]$RngSeeds2 = @(42),
  [int]$TotalSteps2 = 100000
)

$TOPDIR = (Resolve-Path $PSScriptRoot).Path
$env:PYTHONHASHSEED = '0'

$chainOut  = Join-Path $TOPDIR 'chain_outputs'
$chainLogs = Join-Path $TOPDIR 'chain_logs'
New-Item -ItemType Directory -Force -Path $chainOut,$chainLogs | Out-Null

foreach ($seed in $RngSeeds) {
  $outFile = Join-Path $chainOut  "gerrymandria_chain_${TotalSteps}_steps_seed$seed.jsonl"
  $logFile = Join-Path $chainLogs "log_simple_rng_seed_$seed.log"

  & uv run (Join-Path "$TOPDIR" (Join-Path "pipeline_scripts" "example_cli.py")) `
    --graph-path   (Join-Path "$TOPDIR" (Join-Path "JSON_dualgraphs" "gerrymandria.json")) `
    --output-path  "$outFile" `
    --starting-plan "district" `
    --pop-col       "TOTPOP" `
    --rng-seed      $seed `
    --population-tolerance 0.01 `
    --total-steps   $TotalSteps `
    --writeas "jsonl" *> $logFile
}

foreach ($seed in $RngSeeds2) {
  $outFile = Join-Path (Join-Path $TOPDIR "chain_outputs") ("MN_chain_{0}_steps_seed{1}.jsonl.ben" -f $TotalSteps2, $seed)

  & uv run (Join-Path $TOPDIR (Join-Path "pipeline_scripts" "example_cli.py")) `
    --graph-path   (Join-Path $TOPDIR (Join-Path "JSON_dualgraphs" "MN_precincts.geojson")) `
    --output-path  $outFile `
    --starting-plan "CONGDIST" `
    --pop-col       "TOTPOP" `
    --rng-seed      $seed `
    --population-tolerance 0.05 `
    --total-steps   $TotalSteps2 `
    --writeas "ben"
}
'@
'chain_outputs/ben_to_xben.ps1' = @'
param([switch]$Recurse = $true)

# This script converts every BEN file next to it to an XBEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

$files = Get-ChildItem -Path $PSScriptRoot -File -Filter *.ben -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  # -c -1 lets the XZ encoder use every available core
  & ben xencode $f.FullName -v -w -c -1
}
'@
'chain_outputs/jsonl_to_ben.ps1' = @'
param([switch]$Recurse = $true)

# This script converts every JSONL file next to it to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

$files = Get-ChildItem -Path $PSScriptRoot -File -Filter *.jsonl -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  & ben encode $f.FullName -v -w
}
'@
'pipeline_scripts/example_cli.py' = @'
from gerrychain import Graph, Partition, MarkovChain
from gerrychain.updaters import Tally
from gerrychain.accept import always_accept
from gerrychain.proposals.tree_proposals import recom
from functools import partial
import random
import jsonlines as jl
import click
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenEncoder
import sys


@click.command()
@click.option("--graph-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", type=click.Path(writable=True, dir_okay=False))
@click.option("--starting-plan", type=str)
@click.option("--pop-col", type=str)
@click.option("--rng-seed", type=int)
@click.option("--population-tolerance", type=float, default=0.01)
@click.option("--total-steps", type=int, default=10_000)
@click.option("--writeas", type=click.Choice(["jsonl", "ben"]), default="ben")
def main(
    graph_path,
    output_path,
    starting_plan,
    pop_col,
    rng_seed,
    population_tolerance,
    total_steps,
    writeas,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    try:
        if graph_path.endswith(".json"):
            graph = Graph.from_json(graph_path)
        else:
            graph = Graph.from_file(graph_path)
    except Exception as e:
        raise ValueError(f"Failed to load graph from {graph_path}: {e}")

    initial_partition = Partition(
        graph,
        assignment=starting_plan,
        updaters={"population": Tally(pop_col, alias="population")},
    )

    ideal_pop = sum(initial_partition["population"].values()) / len(initial_partition)

    proposal = partial(
        recom,
        pop_col=pop_col,
        pop_target=ideal_pop,
        epsilon=population_tolerance,
        node_repeats=1,
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[],
        initial_state=initial_partition,
        total_steps=total_steps,
        accept=always_accept,
    )

    graph_node_order = list(graph.nodes)

    # This will print to the standard error stream so that logging does not interfere with the
    # standard output.
    print(
        f"Writing output to '{Path(output_path).name}' in '{writeas.upper()}' format.",
        file=sys.stderr,
        flush=True,
    )
    match writeas:
        case "jsonl":
            with jl.open(output_path, "w") as writer:
                for i, partition in enumerate(chain.with_progress_bar()):
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    writer.write(
                        {
                            "assignment": ordered_assignment,
                            "sample": i + 1,
                        }
                    )

        case "ben":
            with BenEncoder(output_path, overwrite=True) as encoder:
                for partition in chain.with_progress_bar():
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    encoder.write(ordered_assignment)

        case _:
            raise ValueError(f"Unsupported writeas format: {writeas}")


if __name__ == "__main__":
    main()
'@
'pipeline_scripts/metrics/process_partisan_bias.py' = @'
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from pathlib import Path
import geopandas as gpd
import numpy as np
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, vote_arrays):
    assign = np.asarray(assignment_vector, dtype=np.int64)

    k = int(assign.max())  # number of districts

    out = {"sample": sample_idx, "pb_scores": {}}

    for dem_votes, rep_votes, name in vote_arrays:
        # assignments are 1-indexed, so drop bin 0 to keep the phantom empty
        # district out of mean_share
        dem_tot = np.bincount(assign, weights=dem_votes, minlength=k + 1)[1:]
        rep_tot = np.bincount(assign, weights=rep_votes, minlength=k + 1)[1:]

        total = dem_tot + rep_tot
        dem_share = np.divide(
            dem_tot, total, out=np.zeros_like(dem_tot, dtype="float64"), where=total > 0
        )

        mean_share = dem_share.mean()
        pb = (dem_share > mean_share).sum() / k - 0.5
        out["pb_scores"][name] = float(pb)

    return out


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_partisan_bias_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    gdf = gpd.read_file(GRAPH_PATH)

    elections = ["PRES16", "SSEN16"]
    election_pairs = [(f"{name}D", f"{name}R") for name in elections]

    # grab vote columns as numpy arrays once
    vote_arrays = []
    for d_col, r_col in election_pairs:
        vote_arrays.append(
            (
                gdf[d_col].to_numpy(dtype="float64", copy=True),
                gdf[r_col].to_numpy(dtype="float64", copy=True),
                d_col[:-1],
            )
        )
    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing partisan bias (batch {batch_no+1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, vote_arrays) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
'pipeline_scripts/metrics/process_polsby.py' = @'
import jsonlines as jl
from gerrychain import GeographicPartition, Graph
from gerrychain.metrics import polsby_popper
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_number, assignment_vector, graph):
    part = GeographicPartition(
        graph, assignment={i: val for i, val in enumerate(assignment_vector)}
    )
    return {"sample": sample_number, "scores": polsby_popper(part)}


if __name__ == "__main__":
    batch_size = 1000
    n_samples = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_polsby_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

    np.random.seed(42)  # seed so the subsample is reproducible
    subsamples = sorted(
        map(
            int, np.random.choice(total_chain_length, size=n_samples, replace=False) + 1
        )
    )  # +1 for 1-based indexing

    graph = Graph.from_file(GRAPH_PATH)

    all_scores = []
    n_batches = (len(subsamples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = subsamples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all Polsby-Popper scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, graph) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
'pipeline_scripts/metrics/process_reock.py' = @'
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import geopandas as gpd
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, geo_only):
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="pygeos support was removed in 1.0. geopandas.use_pygeos is a no-op",
        category=UserWarning,
    )

    from gerrytools.scoring import reock

    geo_new = geo_only.copy()
    geo_new["assignment"] = np.array(assignment_vector) - 1
    dissolved = geo_new.dissolve(by="assignment")
    return {"sample": sample_idx, "scores": reock().apply(dissolved)}


if __name__ == "__main__":
    batch_size = 1000
    n_samples = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_reock_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

    np.random.seed(42)  # seed so the subsample is reproducible
    subsamples = sorted(
        map(
            int, np.random.choice(total_chain_length, size=n_samples, replace=False) + 1
        )
    )  # +1 for 1-based indexing

    gdf = gpd.read_file(GRAPH_PATH)
    geo_only = gdf[["geometry"]]

    all_scores = []
    n_batches = (len(subsamples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = subsamples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all Reock scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, geo_only) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
'pipeline_scripts/metrics/process_splits.py' = @'
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    u,
    v,
    same_county_mask,
    county_names,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    cut = assignment[u] != assignment[v]
    cut_edges = int(cut.sum())
    county_splits = len(set(county_names[cut & same_county_mask]))
    return {
        "sample": sample_idx,
        "scores": {
            "county_splits": county_splits,
            "cut_edges": cut_edges,
        },
    }


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_split_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    graph = Graph.from_file(GRAPH_PATH)
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    u = edges[:, 0]
    v = edges[:, 1]

    # Masks that depend only on the graph (not on assignments)
    # A "split" counts only when it's a cut edge AND both endpoints share the same county/place.
    same_county_mask = np.fromiter(
        (
            graph.nodes[int(a)]["COUNTYNAME"] == graph.nodes[int(b)]["COUNTYNAME"]
            for a, b in edges
        ),
        dtype=bool,
        count=len(edges),
    )

    # Don't need to record b since we are going to filter to when it has the same value as a
    county_names = np.fromiter(
        (graph.nodes[int(a)]["COUNTYNAME"] for a, _ in edges),
        dtype=object,
        count=len(edges),
    )

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all split scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(idx, vec, u, v, same_county_mask, county_names)
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
'pipeline_scripts/metrics/process_total_dem_wins.py' = @'
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import geopandas as gpd
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    dem_count_matrix,
    rep_count_matrix,
    race_names,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    race_totals = {name: 0 for name in race_names}

    for part in np.unique(assignment):
        mask = assignment == part
        dem_totals = dem_count_matrix[mask].sum(axis=0)
        rep_totals = rep_count_matrix[mask].sum(axis=0)
        dem_wins = dem_totals > rep_totals

        for i, race in enumerate(race_names):
            race_totals[race] += 1 if dem_wins[i] else 0

    return {"sample": sample_idx, "scores": race_totals}


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_dem_win_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    df = gpd.read_file(GRAPH_PATH)
    graph = Graph.from_geodataframe(df)
    df.drop(columns=["geometry"], inplace=True)

    race_names = [
        "PRES16",
        "SSEN16",
    ]

    dem_rep_pairs = [(f"{name}D", f"{name}R") for name in race_names]
    dem_count_matrix = df[[pair[0] for pair in dem_rep_pairs]].to_numpy()
    rep_count_matrix = df[[pair[1] for pair in dem_rep_pairs]].to_numpy()

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing total dem wins in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(
                    idx, vec, dem_count_matrix, rep_count_matrix, race_names
                )
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
'pipeline_scripts/rust_example_script.ps1' = @'
param(
  [string]$PlanName = 'district',
  [int]$n_steps = 1000,
  [int]$seed = 42,
  [double]$tol = 0.01,
  [string]$pop_col = 'TOTPOP'
)

# Project root is the parent of this script's folder
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$json_dir   = Join-Path $ProjectRoot 'JSON_dualgraphs'
$output_dir = Join-Path $ProjectRoot 'chain_outputs'

# Ensure output dir exists
New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

# Build JSON path (don't Resolve-Path until we know it exists)
$json_file = Join-Path $json_dir 'gerrymandria.json'

if (-not (Test-Path $json_file)) {
  Write-Error "Could not find graph JSON at: $json_file`nDid the bootstrap step download it?"
  exit 1
}

$final_output_file = Join-Path $output_dir ("gerrymandria_chain_{0}_steps.jsonl.ben" -f $n_steps)

& frcw `
  --assignment-col $PlanName `
  --graph-json $json_file `
  --n-steps $n_steps `
  --pop-col $pop_col `
  --rng-seed $seed `
  --tol $tol `
  --variant district-pairs-rmst `
  --writer ben `
  --batch-size 1 `
  --n-threads 1 `
  --output-file $final_output_file
'@
'pyproject.toml' = @'
[project]
name = "redistricting-project"
version = "0.1.0"
description = "A ready-to-run redistricting analysis project"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "binary-ensemble>=1.0",
    "click",
    "docker",
    "gerrychain[geo]",
    "gerrytools",
    "ipykernel",
    "ipywidgets",
    "joblib",
    "joblib-progress",
    "jsonlines",
    "matplotlib",
    "maup",
    "numpy",
    "pandas",
    "seaborn",
]

[dependency-groups]
dev = [
    "black",
]
'@
}

# Writes every embedded project file into the current (project) directory.
function Write-PayloadFiles
{
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    foreach ($rel in $PayloadDirectories)
    {
        New-Item -ItemType Directory -Force -Path $rel | Out-Null
    }
    foreach ($rel in $Payloads.Keys)
    {
        $destDir = Split-Path -Path $rel -Parent
        if (-not [string]::IsNullOrWhiteSpace($destDir))
        {
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        }
        # BOM-less UTF-8: PS5's Out-File -Encoding UTF8 writes a BOM, which
        # e.g. json parsers reject
        $dest = Join-Path (Get-Location).Path $rel
        [IO.File]::WriteAllText($dest, $Payloads[$rel] + "`n", $utf8NoBom)
    }
}

# ==============================================
# ========  MAIN INSTALLATION FUNCTION  ========
# ==============================================

function Main
{
    Confirm-Uv

    $projectName = Read-Host "Enter the name of the new project to create"
    if ([string]::IsNullOrWhiteSpace($projectName))
    {
        $projectName = "my_project"
        Write-Warn "No project name provided. Using default: $projectName"
    }

    $useFrcw = Read-Host "Would you like to use FRCW in this project? (y/[n])"
    if ($useFrcw -match '^(y|Y)$')
    {
        Confirm-BuildTools
        Confirm-Cargo
        Write-Info "Installing FRCW (rustrecom, branch 0.1.4)..."
        & cargo install --git "https://github.com/mggg/rustrecom" --branch "0.1.4" --force
        Write-OK "FRCW installed."
        Write-Info "Installing binary-ensemble..."
        & cargo install binary-ensemble --force
        Write-OK "binary-ensemble installed."
        Write-Info "Installing ben-process (metrics engine)..."
        & cargo install --git "https://github.com/peterrrock2/ben-process" --force
        Write-OK "ben-process installed."
    } else
    {
        $ans = Read-Host "Would you like to use BEN in this project? (y/[n])"
        if ($ans -match '^(y|Y)$')
        {
            Confirm-Cargo
            Write-Info "Installing binary-ensemble..."
            & cargo install binary-ensemble --force
            Write-OK "binary-ensemble installed."
            Write-Info "Installing ben-process (metrics engine)..."
            & cargo install --git "https://github.com/peterrrock2/ben-process" --force
            Write-OK "ben-process installed."
        }
    }

    $pythonVersion = Read-Host "What python version would you like (3.11, 3.12, 3.13)? (default: 3.11)"
    if ($pythonVersion -notmatch '^(3\.11|3\.12|3\.13)$')
    {
        Write-Warn "Invalid python version. Using default 3.11."
        $pythonVersion = '3.11'
    }

    Write-Info "Creating project: $projectName"
    New-Item -ItemType Directory -Force -Path $projectName | Out-Null
    Push-Location $projectName

    & uv python install $pythonVersion

    Write-Info "Writing project files..."
    Write-PayloadFiles
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText(".python-version", "$pythonVersion`n", $utf8NoBom)
    $pyproject = [IO.File]::ReadAllText("pyproject.toml")
    $pyproject = $pyproject -replace '(?m)^requires-python = .+$', `
        "requires-python = `">=$pythonVersion`""
    [IO.File]::WriteAllText("pyproject.toml", $pyproject, $utf8NoBom)

    Write-Info "Installing the project environment with uv ($pythonVersion)..."
    & uv sync --python $pythonVersion

    Write-Info "Downloading MN_precincts.geojson..."
    $destDir = "JSON_dualgraphs"
    $uri = "https://github.com/mggg/GerryChain/raw/main/docs/_static/MN.zip"

    Invoke-WithRetry -MaxAttempts 5 -Action {

        # create a temp *zip* path (PS5 Expand-Archive checks extension)
        $tmpZip = Join-Path $env:TEMP ("MN_" + [guid]::NewGuid().ToString() + ".zip")

        try
        {
            Invoke-WebRequest -Uri $uri -OutFile $tmpZip -UseBasicParsing
            Expand-Archive -LiteralPath $tmpZip -DestinationPath $destDir -Force
        } finally
        {
            Remove-Item -LiteralPath $tmpZip -ErrorAction SilentlyContinue
        }
    }

    Write-OK "Your project is ready!"
    Write-Warn "If 'uv' or 'cargo' commands are not recognized in *new* shells, log out/in or ensure these are on PATH:"
    Write-Host "  $HOME\.local\bin"
    Write-Host "  $HOME\.cargo\bin"
    Pop-Location
}


if ($MyInvocation.InvocationName -ne '.')
{
    Main
}
