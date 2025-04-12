# Family Photo Manager 🖼️✨

**An AI-powered photo organization system** that lets you search and manage your personal photo collection using natural language queries and visual similarity.

## 🌟 Key Features

- **Natural Language Search** - Find photos with text queries like "beach sunset" or "birthday party 2023"
- **Visual Similarity** - Upload an image to find visually similar photos
- **Bulk Operations** - Rotate or delete multiple photos at once
- **Smart Thumbnails** - Fast browsing with automatically generated previews
- **Cross-Platform** - Works on Windows, macOS, and Linux
- **Privacy Focused** - All processing happens locally on your machine

## 🛠️ Technical Highlights

- Powered by OpenAI's CLIP model for state-of-the-art image understanding
- Uses PyTorch for efficient deep learning operations
- SQLite database for fast feature storage and retrieval
- Flask-based web interface for easy access
- Background processing for smooth user experience

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
pip install --upgrade pip setuptools wheel

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

# 8. Process your photos to generate features
python image_features_manager.py --photos path_to_your_photos

# 9. Launch the application
python app.py --photos path_to_your_photos --port 5000
```

## 🖥️ Using the Application

Once running, access the web interface at `http://localhost:5000`

### Core Functions:
1. **Search**:
   - Type natural language descriptions in the search box
   - Or drag & drop an image to find similar photos

2. **Manage**:
   - Select multiple photos (Ctrl+Click or Cmd+Click)
   - Use the bottom toolbar to rotate or delete selected photos
   - Click thumbnails to view full-resolution images

3. **Browse**:
   - Scroll to load more results
   - Use pagination controls for large collections

## ⚙️ Advanced Configuration

### Custom Python Executable
If you have multiple Python versions:
```bash
./setup.sh --photos your_photos --python /path/to/python.exe
```

### Custom Port
To run on a different port:
```bash
./setup.sh --photos your_photos --port 8000
```

### Force Reinstall
To recreate the virtual environment:
```bash
./setup.sh --photos your_photos --force
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'flask'" | Activate virtual environment first |
| PyTorch installation fails | Use Python 3.8-3.10 or specify `--python` |
| "Access is denied" on Windows | Run terminal as Administrator |
| CUDA not detected | Install NVIDIA drivers or use CPU-only mode |
| Slow processing | Reduce batch size in image_features_manager.py |

## 📂 Project Structure

```
family_photo_manager/
├── setup.sh                  # Automated setup script
├── app.py                    # Main Flask application
├── image_features_manager.py # Feature extraction
├── requirements.txt          # Dependency list
├── helpers/                  # Test scripts and others
├── templates/                # HTML templates
├── tmp/                      # Database and cache
└── helpers/                  # Utility functions
```

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