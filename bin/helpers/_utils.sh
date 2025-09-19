#!/bin/bash
# Shared helper functions for scripts
# Source this file in other scripts with: source "$(dirname "$0")/helpers"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set source and root directory
set_source_and_root_dir() {
	SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	ROOT_DIR="$(cd "$SOURCE_DIR/../.." && pwd)"
	cd "$ROOT_DIR" || exit 1
}

# Color printing functions
print_color() {
	local color=$1
	shift
	case $color in
	red) echo -e "${RED}$*${NC}" ;;
	green) echo -e "${GREEN}$*${NC}" ;;
	blue) echo -e "${BLUE}$*${NC}" ;;
	yellow) echo -e "${YELLOW}$*${NC}" ;;
	*) echo "$*" ;;
	esac
}

# Logging functions
info() {
	print_color blue "ℹ $1"
}

success() {
	print_color green "✓ $1"
}

warning() {
	print_color yellow "⚠ $1"
}

error() {
	print_color red "✗ $1"
}

# Show help function
show_help() {
	local script_name
	script_name="$(basename "$0")"
	# Use BASH_SOURCE to get the actual script path
	if [ -n "${BASH_SOURCE[1]}" ]; then
		grep '^#/' "${BASH_SOURCE[1]}" | sed 's|^#/||g' | sed "s|\$0|$script_name|g"
	else
		grep '^#/' "$0" | sed 's|^#/||g' | sed "s|\$0|$script_name|g"
	fi
}

# Check if command exists
command_exists() {
	command -v "$1" >/dev/null 2>&1
}

# Run command with proper error handling and logging
run_command() {
	local cmd="$1"
	local desc="$2"
	local verbose="${VERBOSE:-0}"

	if [ -n "$desc" ]; then
		if [ "$verbose" = "1" ]; then
			print_color blue "Running: $desc"
			print_color blue "Command: $cmd"
		else
			print_color blue "$desc"
		fi
	fi

	if [ "$verbose" = "1" ]; then
		eval "$cmd"
	else
		eval "$cmd" >/dev/null 2>&1
	fi

	local exit_code=$?
	if [ $exit_code -eq 0 ]; then
		if [ -n "$desc" ]; then
			print_color green "✓ $desc"
		fi
		return 0
	else
		if [ -n "$desc" ]; then
			print_color red "✗ $desc failed"
		fi
		return 1
	fi
}

# Check if we're in a Python virtual environment
check_venv() {
	if [ -z "$VIRTUAL_ENV" ]; then
		warning "Not in a Python virtual environment"
		info "Consider running: python -m venv venv && source venv/bin/activate"
		return 1
	fi
	return 0
}

# Check for existing virtual environment (doesn't activate)
find_venv() {
	local venv_paths=("venv" ".venv" "env" ".env")

	for venv_path in "${venv_paths[@]}"; do
		if [ -f "$venv_path/bin/activate" ]; then
			echo "$venv_path"
			return 0
		fi
	done

	return 1
}

# Create and/or activate virtual environment
ensure_venv() {
	# If already in venv, we're good
	if [ -n "$VIRTUAL_ENV" ]; then
		return 0
	fi

	# Check for existing venv
	local existing_venv
	if existing_venv=$(find_venv); then
		error "Found virtual environment at '$existing_venv' but it's not activated"
		info "Please activate it and run this command again:"
		print_color green "  source $existing_venv/bin/activate"
		print_color green "  $0 $*"
		return 2 # Special return code for "please activate and retry"
	fi

	# No venv found, create one
	info "No virtual environment found, creating one..."
	if ! run_command "python -m venv venv" "Creating virtual environment"; then
		return 1
	fi

	print_color blue "Virtual environment created!"
	print_color yellow "Please activate it and run this command again:"
	print_color green "  source venv/bin/activate  # Unix/macOS"
	print_color green "  venv\\Scripts\\activate     # Windows"
	print_color green "  $0 $*"
	return 2 # Special return code for "please activate and retry"
}

# Check for required Python dependencies
check_python_deps() {
	local missing_deps=()

	# Check for basic Python tools
	if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
		missing_deps+=("Python 3.8+")
	fi

	# Check for Python 3.11 alpha versions that are missing tomllib
	if python -c "import sys; sys.exit(0 if sys.version_info[:3] == (3, 11, 0) and 'a' in sys.version else 1)" 2>/dev/null; then
		warning "Detected Python 3.11.0 alpha version"
		warning "Python 3.11.0a2 is missing the 'tomllib' module required by modern pip"
		info "This will cause pip upgrade and dependency installation failures"
		info "Recommended solutions:"
		info "  1. Use Python 3.11.0 final or later"
		info "  2. Use Python 3.10.x"
		info "  3. Use Python 3.12+"
		info "Continuing with limited functionality..."
		return 2 # Special return code for "compatible but problematic"
	fi

	if [ ${#missing_deps[@]} -gt 0 ]; then
		error "Missing required dependencies:"
		for dep in "${missing_deps[@]}"; do
			echo "  - $dep"
		done
		return 1
	fi

	return 0
}

# Check if we're in the project root
check_project_root() {
	if [ ! -f "requirements-lite.txt" ] || [ ! -d "src/spelungit" ]; then
		error "Please run this script from the project root directory"
		exit 1
	fi
}

# Install dependencies if needed
ensure_deps() {
	local req_file="$1"
	if [ ! -f "$req_file" ]; then
		error "Requirements file $req_file not found"
		exit 1
	fi

	info "Ensuring dependencies from $req_file are installed..."
	pip install -r "$req_file" --quiet
}

# Set PYTHONPATH for the project
setup_pythonpath() {
	export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
}

# Common setup for most scripts
common_setup() {
	check_project_root
	setup_pythonpath
}

# Check if Git repository is clean
check_git_clean() {
	if ! git diff-index --quiet HEAD --; then
		warning "Git repository has uncommitted changes"
		return 1
	fi
	return 0
}

# Get project version (placeholder for now)
get_version() {
	echo "dev"
}

# Check if we can connect to PyPI (for offline detection)
check_pypi_connectivity() {
	# Try a quick connection test to PyPI
	if timeout 5 python -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5)" >/dev/null 2>&1; then
		return 0
	else
		return 1
	fi
}

# Install Python package with better error handling
install_package() {
	local requirements_file="$1"
	local description="$2"

	# First check if we can connect to PyPI
	if ! check_pypi_connectivity; then
		warning "Cannot connect to PyPI - you may be offline or have network issues"
		info "Skipping package installation for now"
		return 1
	fi

	# Try normal installation
	if pip install -r "$requirements_file" >/dev/null 2>&1; then
		success "$description"
		return 0
	else
		error "$description failed"
		info "This may be due to network connectivity or SSL certificate issues"
		return 1
	fi
}

# Attempt to upgrade pip using get-pip.py bootstrap (for very old pip versions)
upgrade_old_pip() {
	info "Attempting to upgrade old pip using get-pip.py bootstrap script..."

	# Try to download and run get-pip.py
	if command_exists curl; then
		if curl -sSL https://bootstrap.pypa.io/get-pip.py | python; then
			info "Successfully upgraded pip using get-pip.py"
			return 0
		fi
	elif command_exists wget; then
		if wget -qO- https://bootstrap.pypa.io/get-pip.py | python; then
			info "Successfully upgraded pip using get-pip.py"
			return 0
		fi
	fi

	return 1
}
