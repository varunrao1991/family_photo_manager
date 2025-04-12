#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PHOTO_DIR=""
VENV_DIR="venv"
PORT=5000
FORCE_REINSTALL=false
PYTHON_COMMAND=""
IS_WINDOWS=false

# Detect Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OS" == "Windows_NT" ]]; then
    IS_WINDOWS=true
fi

show_help() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./setup.sh [options]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --photos <dir>    Set photo directory (required)"
    echo "  --venv <dir>      Set virtual environment directory (default: venv)"
    echo "  --port <number>   Set application port (default: 5000)"
    echo "  --python <path>   Specify full path to Python executable"
    echo "  --force           Force reinstall of all packages"
    echo "  --help            Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --photos) PHOTO_DIR="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --python) PYTHON_COMMAND="$2"; shift 2 ;;
        --force) FORCE_REINSTALL=true; shift ;;
        --help) show_help ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; show_help ;;
    esac
done

if [ -z "$PHOTO_DIR" ]; then
    echo -e "${RED}Error: --photos <dir> is required${NC}"
    show_help
fi

check_python() {
    if [ -n "$PYTHON_COMMAND" ]; then
        if [ -x "$PYTHON_COMMAND" ] || command -v "$PYTHON_COMMAND" &> /dev/null; then
            PYTHON_COMMAND=$(command -v "$PYTHON_COMMAND")
        else
            echo -e "${RED}Specified Python not found: $PYTHON_COMMAND${NC}"
            return 1
        fi
    else
        # Windows-specific Python locations
        if [ "$IS_WINDOWS" = true ]; then
            # Check common Windows Python locations
            declare -a python_paths=(
                "$LOCALAPPDATA/Programs/Python/Python310/python.exe"
                "$LOCALAPPDATA/Programs/Python/Python39/python.exe"
                "$LOCALAPPDATA/Programs/Python/Python38/python.exe"
                "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python310/python.exe"
                "/c/Program Files/Python310/python.exe"
            )
            
            for python_path in "${python_paths[@]}"; do
                if [ -f "$python_path" ]; then
                    PYTHON_COMMAND="$python_path"
                    break
                fi
            done
        fi

        # Fall back to standard python commands
        if [ -z "$PYTHON_COMMAND" ]; then
            for candidate in python3 python python3.10 python3.9 python3.8; do
                if command -v "$candidate" &> /dev/null; then
                    VERSION=$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
                    if [[ "$VERSION" =~ ^3\.(8|9|10)$ ]]; then
                        PYTHON_COMMAND="$candidate"
                        break
                    fi
                fi
            done
        fi
    fi

    if [ -z "$PYTHON_COMMAND" ]; then
        echo -e "${RED}No suitable Python 3.8–3.10 found${NC}"
        if [ "$IS_WINDOWS" = true ]; then
            echo -e "${YELLOW}On Windows, try installing Python from python.org and specify path with --python${NC}"
        fi
        return 1
    fi

    PYTHON_VERSION=$("$PYTHON_COMMAND" --version 2>&1)
    echo -e "${GREEN}Using Python: $PYTHON_COMMAND ($PYTHON_VERSION)${NC}"
    return 0
}

setup_venv() {
    if [ -d "$VENV_DIR" ] && [ "$FORCE_REINSTALL" = false ]; then
        echo -e "${YELLOW}Using existing virtual environment: $VENV_DIR${NC}"
        return
    fi

    # Clean up existing venv if forcing reinstall
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    fi

    echo -e "${YELLOW}Creating virtual environment in $VENV_DIR...${NC}"
    "$PYTHON_COMMAND" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment${NC}"
        exit 1
    fi

    # Windows-specific fix for pip
    if [ "$IS_WINDOWS" = true ]; then
        echo -e "${YELLOW}Applying Windows-specific fixes...${NC}"
        # Ensure pip is properly initialized
        "$PYTHON_COMMAND" -m ensurepip --upgrade
        "$PYTHON_COMMAND" -m pip install --upgrade pip
    fi
}

activate_venv() {
    if [ "$IS_WINDOWS" = true ]; then
        ACTIVATE="$VENV_DIR/Scripts/activate"
    else
        ACTIVATE="$VENV_DIR/bin/activate"
    fi

    if [ -f "$ACTIVATE" ]; then
        source "$ACTIVATE"
        echo -e "${GREEN}Virtual environment activated${NC}"
        
        # Verify pip is working
        if ! python -m pip --version &> /dev/null; then
            echo -e "${RED}Pip is not working in the virtual environment${NC}"
            echo -e "${YELLOW}Attempting to repair pip...${NC}"
            python -m ensurepip --upgrade
            python -m pip install --upgrade pip
            if ! python -m pip --version &> /dev/null; then
                echo -e "${RED}Failed to repair pip${NC}"
                exit 1
            fi
        fi
    else
        echo -e "${RED}Activation script not found: $ACTIVATE${NC}"
        exit 1
    fi
}

install_deps() {
    echo -e "${YELLOW}Installing dependencies...${NC}"

    for i in {1..3}; do
        python -m pip install --upgrade pip && break || {
            echo -e "${YELLOW}Pip upgrade failed (attempt $i/3), retrying...${NC}"
            sleep 1
            [ "$i" -eq 3 ] && { echo -e "${RED}Failed to upgrade pip${NC}"; exit 1; }
        }
    done

    PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
    if [[ "$PYTHON_MINOR" -ge 11 ]]; then
        echo -e "${YELLOW}Detected Python 3.11+, installing compatible PyTorch...${NC}"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        echo -e "${YELLOW}Installing PyTorch 1.13.1 for Python 3.8-3.10...${NC}"
        pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
    fi

    if [ -f "requirements.txt" ]; then
        grep -v 'git+https://github.com/openai/CLIP.git' requirements.txt > requirements-clean.txt
        pip install -r requirements-clean.txt
        rm requirements-clean.txt
    fi

    if [ ! -d "external/clip" ]; then
        echo -e "${YELLOW}Cloning OpenAI CLIP...${NC}"
        git clone https://github.com/openai/CLIP.git external/clip
    else
        echo -e "${YELLOW}Using existing CLIP repo in external/clip${NC}"
    fi

    pip install -e external/clip

    echo -e "${GREEN}Dependencies installed${NC}"
}

prepare_images() {
    echo -e "${YELLOW}Prepare images with photo dir: $PHOTO_DIR${NC}"
    python image_features_manager.py --photos "$PHOTO_DIR"
}

launch_app() {
    echo -e "${YELLOW}Launching app on port $PORT with photo dir: $PHOTO_DIR${NC}"
    python app.py --photos "$PHOTO_DIR" --port "$PORT"
}

main() {
    echo -e "\n${GREEN}=== Family Photo Manager Setup ===${NC}\n"
    check_python || exit 1
    setup_venv
    activate_venv
    install_deps
    prepare_images
    launch_app
}

main