# Family Photo TV

A Samsung Tizen TV web app frontend for the existing Flask photo-search backend.

## What it does

- Sends typed natural-language searches to `POST /similar-images`.
- Displays photos and sampled video frames together in the search results. Video tiles have a centered play icon.
- Opens photos full screen and plays videos with the TV's native video controls; left/right moves through the ordered matches.
- Offers a voice-search button using the browser Speech Recognition API when the TV firmware exposes it. If not, the app explains that the remote keyboard can be used instead.
- Lets you change the photo server URL from the TV screen.

## Run with the current Python backend

1. Start the existing backend with `setup.sh` and `D:\images` as the photo directory (or launch `app.py --photos D:\images`).
2. Keep the laptop and TV on the same Wi-Fi network. The laptop's current address is `192.168.1.4`, so the TV screen starts with `http://192.168.1.4:5000`; edit it if the laptop address changes.
3. Build and install this Tizen project from `FamilyPhotoTV` with a Samsung TV signing profile configured in Tizen Studio.

The Flask app adds permissive CORS response headers so the Tizen app can request search results and image files from the laptop. Keep the Flask server on a trusted home network.

`setup.sh --photos D:\images` invokes `image_features_manager.py`. The indexer keeps the existing photo embeddings and adds one CLIP embedding for a video frame at 0, 120, 240 seconds, and so on. Video embeddings are stored separately in `video_embeddings`; each video appears once in search, ranked by its best matching frame. Install FFmpeg and ffprobe and add them to `PATH` for video indexing. Playback streams a source file if it is compatible; otherwise the backend converts it on the first play request and caches the H.264/AAC MP4 under `VIDEO_CONVERTED_DIR` (default `tmp/converted-videos`).

## Project

Open `FamilyPhotoTV` in Tizen Studio. It uses the Samsung TV profile. Signing and deployment require a certificate profile created with Samsung Certificate Extension. The TV was reachable from the development laptop at `192.168.1.12:26101` when this prototype was started.
