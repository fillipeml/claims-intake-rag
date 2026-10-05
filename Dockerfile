# The service, with the committed fixtures and nothing that needs a GPU.
#
# Two stages so the build tools do not ship: the first resolves and installs into a virtual
# environment, the second copies that environment into a clean image. The result is a few
# hundred megabytes rather than a gigabyte, and the running container holds no compiler.
#
# The [models] extra is deliberately absent. It would pull torch and some gigabytes of model
# weights to do what the committed vectors already do, and the image would stop being something
# anyone can pull and run in a minute. A deployment that must embed live documents builds with
# `--build-arg EXTRAS="serve,models"`.

FROM python:3.12-slim AS build

ARG EXTRAS="serve"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /src

RUN python -m venv /opt/venv

# The manifest first, so a change to the source does not re-resolve the dependencies.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[${EXTRAS}]"


FROM python:3.12-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLAIMS_FIXTURES=/app/fixtures

# Unprivileged, and the files are owned by root so the process cannot rewrite its own code.
RUN useradd --create-home --uid 10001 claims

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY fixtures ./fixtures

USER claims
EXPOSE 8000

# Hits the application rather than the port: a container that is listening but could not load
# its corpus is not healthy, and a TCP check would call it healthy for ever.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker. The work is CPU-bound retrieval over an in-process index, so a second worker in
# the same container would double the memory for the same corpus and contend for the same
# cores. Scale with replicas, not with workers.
CMD ["uvicorn", "claims_rag.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
