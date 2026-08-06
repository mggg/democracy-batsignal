FROM mcr.microsoft.com/powershell:ubuntu-24.04@sha256:042240d57ec9e47e511033b92625a8d95875ee5860af3015992c248b58a8be81 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        build-essential \
        ca-certificates \
        curl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /projects
ENV UV_LINK_MODE=copy

FROM base AS powershell-project

COPY democracy-batsignal.ps1 /installers/
RUN printf '%s\n' y powershell_project y y 3.11 \
    | pwsh -NoProfile -File /installers/democracy-batsignal.ps1

ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

FROM base AS bash-project

COPY democracy-batsignal.sh /installers/
RUN printf '%s\n' y bash_project y y 3.11 \
    | bash /installers/democracy-batsignal.sh

FROM powershell-project AS smoke-test

COPY --from=bash-project /projects/bash_project /projects/bash_project
COPY test_generated_projects.sh /usr/local/bin/test-generated-projects
RUN chmod +x /usr/local/bin/test-generated-projects

ENV BATSIGNAL_IN_CONTAINER=1
ENV MPLBACKEND=Agg

ENTRYPOINT ["/usr/local/bin/test-generated-projects"]
