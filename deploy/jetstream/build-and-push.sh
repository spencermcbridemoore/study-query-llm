#!/usr/bin/env bash
# Build the study-query-llm Docker image from the repo root, tag it with the
# current git commit SHA, and optionally push it to a registry.
#
# Usage:
#   ./build-and-push.sh                # build only (local tag)
#   ./build-and-push.sh --push         # build + push to registry
#   REGISTRY=ghcr.io/yourorg ./build-and-push.sh --push
#
# The script prints the IMAGE_REF you should paste into .env.jetstream.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_NAME="study-query-llm"
REGISTRY="${REGISTRY:-}"
PUSH=false

for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Resolve git commit SHA for the tag.
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short=8 HEAD)"
DATE_TAG="$(date -u +%Y%m%d)"
TAG="${DATE_TAG}-${GIT_SHA}"

if [ -n "$REGISTRY" ]; then
    FULL_REF="${REGISTRY}/${IMAGE_NAME}:${TAG}"
else
    FULL_REF="${IMAGE_NAME}:${TAG}"
fi

echo "==> Building image from ${REPO_ROOT}"
echo "    Tag: ${FULL_REF}"

docker build \
    --target runtime \
    -t "${FULL_REF}" \
    -t "${IMAGE_NAME}:latest" \
    "$REPO_ROOT"

echo ""
echo "==> Build complete."
echo "    IMAGE_REF=${FULL_REF}"

if [ "$PUSH" = true ]; then
    if [ -z "$REGISTRY" ]; then
        echo "ERROR: --push requires REGISTRY env var (e.g. ghcr.io/yourorg)"
        exit 1
    fi
    echo "==> Pushing ${FULL_REF} ..."
    docker push "${FULL_REF}"

    DIGEST="$(docker inspect --format='{{index .RepoDigests 0}}' "${FULL_REF}" 2>/dev/null || true)"
    if [ -n "$DIGEST" ]; then
        echo ""
        echo "==> Digest-pinned ref (preferred for .env.jetstream):"
        echo "    IMAGE_REF=${DIGEST}"
    fi
fi

echo ""
echo "==> Done. Update IMAGE_REF in deploy/jetstream/.env.jetstream with the ref above."
