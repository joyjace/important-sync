#!/usr/bin/env bash
set -euo pipefail

# Copy every file tracked by important-sync into the matching path in the
# local esp-csi get-started tree. Run this as rostami1 or rostami2 (no sudo).
#
# Usage:
#   /home/rostami1/important-sync/apply_important_sync.sh
#   /home/rostami2/important-sync/apply_important_sync.sh
#   ./apply_important_sync.sh --dry-run
#
# Test/non-standard path overrides:
#   PI_SYNC_ROOT=/path/to/important-sync \
#   PI_PROJECT_ROOT=/path/to/get-started ./apply_important_sync.sh

DRY_RUN=0
if [[ ${1:-} == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

DETECTED_PI_USER="${SUDO_USER:-$(id -un)}"
case "$DETECTED_PI_USER" in
    rostami1|rostami2)
        ;;
    *)
        if [[ -z ${PI_SYNC_ROOT:-} || -z ${PI_PROJECT_ROOT:-} ]]; then
            echo "ERROR: Run as rostami1 or rostami2, or set both PI_SYNC_ROOT and PI_PROJECT_ROOT." >&2
            exit 2
        fi
        ;;
esac

PI_SYNC_ROOT="${PI_SYNC_ROOT:-/home/$DETECTED_PI_USER/important-sync}"
PI_PROJECT_ROOT="${PI_PROJECT_ROOT:-/home/$DETECTED_PI_USER/Downloads/esp-csi-master/examples/get-started}"

if [[ ! -d "$PI_SYNC_ROOT/.git" ]]; then
    echo "ERROR: important-sync Git repository not found: $PI_SYNC_ROOT" >&2
    exit 1
fi
if [[ ! -d "$PI_PROJECT_ROOT" ]]; then
    echo "ERROR: destination project directory not found: $PI_PROJECT_ROOT" >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required to enumerate the synced files." >&2
    exit 1
fi

sync_root_resolved="$(realpath "$PI_SYNC_ROOT")"
project_root_resolved="$(realpath "$PI_PROJECT_ROOT")"
if [[ "$sync_root_resolved" == "$project_root_resolved" ]]; then
    echo "ERROR: source and destination resolve to the same directory." >&2
    exit 1
fi
case "$project_root_resolved/" in
    "$sync_root_resolved/"*)
        echo "ERROR: destination must not be inside important-sync." >&2
        exit 1
        ;;
esac
case "$sync_root_resolved/" in
    "$project_root_resolved/"*)
        echo "ERROR: important-sync must not be inside the destination project." >&2
        exit 1
        ;;
esac

temporary_path=""
cleanup_temporary_file() {
    if [[ -n "$temporary_path" ]]; then
        rm -f -- "$temporary_path"
    fi
}
trap cleanup_temporary_file EXIT

copied_count=0
missing_count=0
while IFS= read -r -d '' rel_path; do
    case "$rel_path" in
        ""|/*|..|../*|*/..|*/../*)
            echo "ERROR: unsafe tracked path in important-sync: $rel_path" >&2
            exit 1
            ;;
    esac

    source_path="$PI_SYNC_ROOT/$rel_path"
    destination_path="$PI_PROJECT_ROOT/$rel_path"

    if [[ ! -e "$source_path" && ! -L "$source_path" ]]; then
        echo "WARN: tracked source is missing, skipped: $rel_path" >&2
        missing_count=$((missing_count + 1))
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] $rel_path"
        copied_count=$((copied_count + 1))
        continue
    fi

    destination_dir="$(dirname "$destination_path")"
    mkdir -p -- "$destination_dir"
    destination_dir_resolved="$(realpath "$destination_dir")"
    case "$destination_dir_resolved/" in
        "$project_root_resolved/"*)
            ;;
        *)
            echo "ERROR: destination path escapes the project through a symlink: $rel_path" >&2
            exit 1
            ;;
    esac

    temporary_path="$(mktemp "${destination_path}.important-sync.tmp.XXXXXX")"
    cp -a --remove-destination -- "$source_path" "$temporary_path"
    mv -Tf -- "$temporary_path" "$destination_path"
    temporary_path=""
    echo "[copied] $rel_path"
    copied_count=$((copied_count + 1))
done < <(git -C "$PI_SYNC_ROOT" ls-files -z)

if [[ $copied_count -eq 0 ]]; then
    echo "No tracked files were found in $PI_SYNC_ROOT." >&2
    exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete: $copied_count tracked files would be copied; $missing_count missing."
else
    echo "Restore complete: copied $copied_count tracked files; $missing_count missing."
    echo "Destination: $PI_PROJECT_ROOT"
fi
