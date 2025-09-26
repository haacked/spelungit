#!/bin/bash
set -e

# Git History MCP Server - Zero Config Installation Script
# This script sets up the MCP server with SQLite + local embeddings

echo "🚀 Installing Git History MCP Server (Zero Config Mode)"
echo "=================================================="

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

# Check if we're in the right directory
if [ ! -f "src/spelungit/__init__.py" ]; then
    print_error "Please run this script from the spelungit repository root directory"
    exit 1
fi

# Check Python version
print_info "Checking Python version..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    print_error "Python 3.8+ is required but not found"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

print_status "Python $PYTHON_VERSION found"

# Create virtual environment if it doesn't exist
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_DIR
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
print_info "Installing dependencies (this may take a few minutes for first-time setup)..."
pip install -r requirements-lite.txt > /dev/null 2>&1
print_status "Dependencies installed"

# Install the package in development mode
print_info "Installing spelungit package..."
pip install -e . > /dev/null 2>&1
print_status "Package installed in development mode"

# Create configuration directory
CONFIG_DIR="$HOME/.config/spelungit"
mkdir -p "$CONFIG_DIR"
print_status "Configuration directory created at $CONFIG_DIR"

# Find Claude Code configuration
CLAUDE_CONFIG=""
if [ "$(uname)" = "Darwin" ]; then
    # macOS
    CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
elif [ "$(uname)" = "Linux" ]; then
    # Linux
    CLAUDE_CONFIG="$HOME/.config/claude/claude_desktop_config.json"
else
    print_warning "Unsupported operating system. You'll need to manually configure Claude Code."
fi

# Check if Claude Code config exists
if [ -n "$CLAUDE_CONFIG" ] && [ -f "$CLAUDE_CONFIG" ]; then
    print_info "Found Claude Code configuration at $CLAUDE_CONFIG"

    # Get current directory and Python executable paths
    CURRENT_DIR=$(pwd)
    PYTHON_PATH=$(which python)

    # Create MCP server configuration
    read -p "Do you want to automatically add the MCP server to Claude Code? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Backup existing config
        cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.backup"
        print_status "Backed up existing Claude Code config"

        # Check if config already has mcpServers
        if grep -q '"mcpServers"' "$CLAUDE_CONFIG"; then
            print_info "Adding spelungit to existing mcpServers..."
            # Use Python to safely modify JSON
            $PYTHON_CMD -c "
import json
import os

config_path = '$CLAUDE_CONFIG'
with open(config_path, 'r') as f:
    config = json.load(f)

if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['spelungit'] = {
    'command': '$PYTHON_PATH',
    'args': ['-m', 'spelungit.lite_server'],
    'env': {
        'PYTHONPATH': '$CURRENT_DIR/src'
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('MCP server added to Claude Code configuration')
"
        else
            print_info "Creating new mcpServers configuration..."
            # Create new mcpServers section
            $PYTHON_CMD -c "
import json
import os

config_path = '$CLAUDE_CONFIG'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

config['mcpServers'] = {
    'spelungit': {
        'command': '$PYTHON_PATH',
        'args': ['-m', 'spelungit.lite_server'],
        'env': {
            'PYTHONPATH': '$CURRENT_DIR/src'
        }
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('MCP server configuration created')
"
        fi

        print_status "Git History MCP server added to Claude Code!"
        print_info "Restart Claude Code to load the new MCP server"
    else
        print_info "Skipping automatic Claude Code configuration"
        echo
        print_info "To manually configure Claude Code, add this to your claude_desktop_config.json:"
        echo "{"
        echo "  \"mcpServers\": {"
        echo "    \"spelungit\": {"
        echo "      \"command\": \"$PYTHON_PATH\","
        echo "      \"args\": [\"-m\", \"spelungit.lite_server\"],"
        echo "      \"env\": {"
        echo "        \"PYTHONPATH\": \"$CURRENT_DIR/src\""
        echo "      }"
        echo "    }"
        echo "  }"
        echo "}"
    fi
else
    print_warning "Claude Code configuration not found"
    print_info "You'll need to manually add the MCP server to Claude Code"
    echo
    print_info "Add this to your claude_desktop_config.json:"
    echo "{"
    echo "  \"mcpServers\": {"
    echo "    \"spelungit\": {"
    echo "      \"command\": \"$(which python)\","
    echo "      \"args\": [\"-m\", \"spelungit.lite_server\"],"
    echo "      \"env\": {"
    echo "        \"PYTHONPATH\": \"$(pwd)/src\""
    echo "      }"
    echo "    }"
    echo "  }"
    echo "}"
fi

# Lite server module already exists in src/spelungit/lite_server.py
print_info "Lite server module already present..."

# Test that the package can be imported
if python -c "
import sys
sys.path.insert(0, 'src')
try:
    from spelungit.lite_embeddings import LiteEmbeddingManager
    from spelungit.sqlite_database import SQLiteDatabaseManager
    print('✓ All modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
" 2>/dev/null; then
    print_status "Module imports verified"
else
    print_warning "Module import test failed, but installation may still work"
fi

echo
print_status "Installation completed successfully!"
echo
print_info "🎯 What's next:"
echo "  1. Restart Claude Code to load the MCP server"
echo "  2. Navigate to any Git repository in your terminal"
echo "  3. Start Claude Code and ask: 'What recent changes were made to authentication?'"
echo "  4. The MCP server will automatically detect and index the repository"
echo
print_info "💡 Features available:"
echo "  • Works with any Git repository automatically"
echo "  • No configuration files needed"
echo "  • No external API keys required"
echo "  • Offline-capable with local embeddings"
echo "  • Git worktree support"
echo
print_info "📊 Storage location: $CONFIG_DIR"
print_info "🔧 Virtual environment: $VENV_DIR"
echo
print_status "Ready to explore your Git history with Claude Code! 🚀"
