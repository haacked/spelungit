#!/bin/bash
set -e

# Git History MCP Server - One-Line Remote Installation Script
# Usage: curl -sSL https://raw.githubusercontent.com/haacked/spelungit/main/install-remote.sh | bash

echo "🚀 Installing Spelungit MCP Server"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Cleanup function
cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        print_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi
}

# Set up cleanup on exit
trap cleanup EXIT

# Check prerequisites
print_info "Checking prerequisites..."

# Check Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    print_error "Python 3.8+ is required but not found"
    print_info "Please install Python from https://python.org or your system package manager"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8+ is required (found $PYTHON_VERSION)"
    exit 1
fi

print_status "Python $PYTHON_VERSION found"

# Check download tool
if command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD="curl -sSL"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget -qO-"
else
    print_error "Either curl or wget is required for installation"
    exit 1
fi

# Define installation directory
INSTALL_DIR="$HOME/.local/share/spelungit"
REPO_URL="https://github.com/haacked/spelungit"
ARCHIVE_URL="$REPO_URL/archive/refs/heads/main.tar.gz"

print_info "Installing to: $INSTALL_DIR"

# Create installation directory
mkdir -p "$INSTALL_DIR"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
print_info "Using temporary directory: $TEMP_DIR"

# Download and extract repository
print_info "Downloading Spelungit repository..."
cd "$TEMP_DIR"

# Download with better error handling
if ! $DOWNLOAD_CMD "$ARCHIVE_URL" > archive.tar.gz; then
    print_error "Failed to download repository"
    print_info "Please check your internet connection or try again later"
    print_info "Repository URL: $REPO_URL"
    exit 1
fi

# Extract with better error handling
if ! tar -xzf archive.tar.gz; then
    print_error "Failed to extract repository archive"
    print_info "The downloaded file may be corrupted"
    exit 1
fi

print_status "Repository downloaded and extracted"

# Find the extracted directory (should be spelungit-main)
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "spelungit-*" | head -n1)
if [ -z "$EXTRACTED_DIR" ]; then
    print_error "Could not find extracted repository directory"
    exit 1
fi

cd "$EXTRACTED_DIR"

# Verify we have the expected files
if [ ! -f "install.sh" ] || [ ! -f "src/spelungit/__init__.py" ]; then
    print_error "Downloaded repository appears to be incomplete"
    exit 1
fi

print_status "Repository structure verified"

# Copy to installation directory
print_info "Installing to $INSTALL_DIR..."
cp -r . "$INSTALL_DIR/"

cd "$INSTALL_DIR"

# Make install script executable
chmod +x install.sh

# Run the installation
print_info "Running installation script..."
print_info "This will create a virtual environment and install dependencies..."

if ! ./install.sh; then
    print_error "Installation failed"
    print_info "You can try manual installation by visiting: $REPO_URL"
    exit 1
fi

print_status "Installation completed successfully!"

echo ""
echo "🎉 Spelungit MCP Server is now installed!"
echo ""
print_info "Installation location: $INSTALL_DIR"
print_info "Configuration: Check Claude Code for MCP server setup"

echo ""
print_info "To start using Spelungit:"
echo "  1. Open Claude Code"
echo "  2. Navigate to a Git repository"
echo "  3. Use tools like: index_repository(), search_commits(\"your query\")"

echo ""
print_info "For help and documentation: $REPO_URL"

print_status "Ready to explore your Git history!"
