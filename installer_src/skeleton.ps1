# ---------------------------------------------------------------------------
# This installer is assembled from installer_src/skeleton.ps1 and template_project/.
# Run 'python3 generate_installers.py' after editing either source. Do not edit the
# generated template_maker.ps1 directly.
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

function Assert-NativeSuccess
{
    param([Parameter(Mandatory)][string]$Description)
    if ($LASTEXITCODE -ne 0)
    {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
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
        Write-Err "Cargo is required to install the Rust command-line tools. Exiting."
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

# {{GENERATED_PAYLOADS}}

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

    $useRustReCom = Read-Host "Would you like to use RustReCom in this project? (y/[n])"
    if ($useRustReCom -match '^(y|Y)$')
    {
        Confirm-BuildTools
        Confirm-Cargo
        Write-Info "Installing RustReCom (rustrecom, version 0.2.0)..."
        & cargo install --git "https://github.com/mggg/rustrecom" --tag "v0.2.0" --locked --force
        Assert-NativeSuccess "RustReCom installation"
        Write-OK "RustReCom installed."
    }

    $pythonVersion = Read-Host "What python version would you like (3.11, 3.12, 3.13, 3.14)? (default: 3.11)"
    if ($pythonVersion -notmatch '^(3\.11|3\.12|3\.13|3\.14)$')
    {
        Write-Warn "Invalid python version. Using default 3.11."
        $pythonVersion = '3.11'
    }

    Write-Info "Creating project: $projectName"
    if (Test-Path $projectName)
    {
        Write-Err "A file or directory already exists at '$projectName'. Exiting."
        exit 1
    }
    New-Item -ItemType Directory -Path $projectName | Out-Null
    Push-Location $projectName

    & uv python install $pythonVersion
    Assert-NativeSuccess "Python installation"

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
    Assert-NativeSuccess "Project environment installation"

    Write-Info "Downloading PA geometry..."
    $uri = "https://raw.githubusercontent.com/mggg/democracy-batsignal/" +
        "13c1098d244df946263c9353a478ecf66ac8e484/template_project/data/pa_gdf.parquet"
    $expectedSha256 = "06b3b927b09e3f049623869d0b15e20b1363a2eb4dad43461915382fc165446c"
    $destination = Join-Path "data" "pa_gdf.parquet"

    Invoke-WithRetry -MaxAttempts 5 -Action {
        $tmpFile = [IO.Path]::GetTempFileName()
        try
        {
            Invoke-WebRequest -Uri $uri -OutFile $tmpFile -UseBasicParsing
            $actualSha256 = (Get-FileHash -LiteralPath $tmpFile -Algorithm SHA256).Hash.ToLower()
            if ($actualSha256 -ne $expectedSha256)
            {
                throw "PA data checksum verification failed."
            }
            Move-Item -LiteralPath $tmpFile -Destination $destination -Force
        } finally
        {
            Remove-Item -LiteralPath $tmpFile -ErrorAction SilentlyContinue
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
