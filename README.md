# Family Photo Manager 🖼️✨

**An AI-powered photo organization system** that lets you search and manage your personal photo collection using natural language queries and visual similarity.

## 🌟 Key Features

- **Natural Language Search** - Find photos with text queries like "beach sunset" or "birthday party 2023"
- **Visual Similarity** - Upload an image to find visually similar photos
- **Interactive Image Viewer** - Pan, zoom, and rotate images with intuitive controls
- **Bulk Operations** - Rotate or delete multiple photos at once
- **Smart Thumbnails** - Fast browsing with automatically generated previews
- **Cross-Platform** - Works on Windows, macOS, and Linux
- **Privacy Focused** - All processing happens locally on your machine
- **Environment Config** - Customizable settings via `.env` file

## 🛠️ Technical Highlights

- Powered by OpenAI's CLIP model for state-of-the-art image understanding
- Uses PyTorch for efficient deep learning operations
- SQLite database for fast feature storage and retrieval
- Flask-based web interface for easy access
- Background processing for smooth user experience
- Modern JavaScript frontend with responsive design
- Configurable via environment variables

## 🚀 Quick Start (Automatic Setup)

```bash
# 1. Clone the repository
git clone https://github.com/varunrao1991/family_photo_manager
cd family_photo_manager

# 2. Run the setup script (Windows users - use Git Bash)
./setup.sh --photos path_to_your_photos
```

The automated script will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Process your photos to extract features
4. Launch the web interface at `http://localhost:5000`

## ⚙️ Configuration

Customize the application by creating a `.env` file:

```ini
# Application Configuration
IMAGE_DIR=../images       # Path to your photo directory
PORT=5000                 # Server port
DATABASE=tmp/image_features.db  # Database location
CACHE_DIR=tmp/cache/      # Thumbnail cache directory
MAX_PAGE_SIZE=50          # Maximum results per page
CLIP_MODEL=ViT-B/32       # CLIP model version
DEVICE=cpu                # cpu or auto (auto will use GPU if available)
```

## 📚 Manual Installation (Detailed)

### Requirements
- Python 3.8-3.10 (recommended for PyTorch compatibility)
- pip package manager
- Git (for CLIP installation)
- 4GB+ RAM (8GB recommended for large collections)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/varunrao1991/family_photo_manager
cd family_photo_manager

# 2. Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 3. Install base dependencies
pip install --upgrade pip setuptools wheel python-dotenv

# 4. Install PyTorch with CPU support (recommended for most users)
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# 5. Install other requirements (excluding CLIP)
pip install flask numpy pillow tqdm piexif transformers

# 6. Install CLIP from source
git clone https://github.com/openai/CLIP.git external/clip
pip install -e external/clip

# 7. Prepare your photo directory
mkdir -p tmp/cache  # For thumbnails
mkdir -p your_photos  # Replace with your actual photo directory

# 8. Create .env file (see Configuration section above)

# 9. Process your photos to generate features
python image_features_manager.py --photos path_to_your_photos

# 10. Launch the application
python app.py
```

## 🖥️ Using the Application

Once running, access the web interface at `http://localhost:5000`

### Core Functions:
1. **Search**:
   - Type natural language descriptions in the search box
   - Or drag & drop an image to find similar photos

2. **Image Viewer**:
   - Double-click thumbnails to open full view
   - Drag to pan around zoomed images
   - Mouse near edges to auto-pan large images
   - Use toolbar to rotate or download images

3. **Manage**:
   - Select multiple photos (Ctrl+Click or Cmd+Click)
   - Use the bottom toolbar to rotate or delete selected photos
   - Confirmation dialog for safe deletion

## 📂 Project Structure

```
family_photo_manager/
├── setup.sh                  # Automated setup script
├── app.py                    # Main Flask application
├── image_features_manager.py # Feature extraction
├── requirements.txt          # Dependency list
├── .env                      # Configuration file
├── static/                   # Static assets (CSS, JS)
│   ├── css/
│   └── js/
├── templates/                # HTML templates
├── tmp/                      # Database and cache
└── helpers/                  # Utility functions
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'flask'" | Activate virtual environment first |
| PyTorch installation fails | Use Python 3.8-3.10 or specify `--python` |
| "Access is denied" on Windows | Run terminal as Administrator |
| CUDA not detected | Install NVIDIA drivers or use CPU-only mode |
| Slow processing | Reduce batch size in image_features_manager.py |
| Image viewer issues | Clear browser cache or restart the app |

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## ❤️ Support

For help or feature requests, please:
- [Open an issue](https://github.com/varunrao1991/family_photo_manager/issues)
- Email: varunrao.rao@gmail.com

---

**Happy Photo Organizing!** 📸✨