(() => {
    const queryInput = document.getElementById('query');
    const status = document.getElementById('status');
    const results = document.getElementById('results');
    const viewer = document.getElementById('viewer');
    const viewerFeed = document.getElementById('viewerFeed');
    const viewerHero = document.getElementById('viewerHero');
    const largePhoto = document.getElementById('largePhoto');
    const largeVideo = document.getElementById('largeVideo');
    const videoFeedbackHud = document.getElementById('videoFeedbackHud');
    const videoControls = document.getElementById('videoControls');
    const videoPlayPause = document.getElementById('videoPlayPause');
    const videoRewind = document.getElementById('videoRewind');
    const videoForward = document.getElementById('videoForward');
    const videoCurrentTime = document.getElementById('videoCurrentTime');
    const videoProgressBar = document.getElementById('videoProgressBar');
    const videoDuration = document.getElementById('videoDuration');
    const viewerHelp = document.querySelector('.viewerHelp');
    const photoCaption = document.getElementById('photoCaption');
    const photoName = document.getElementById('photoName');
    const photoDescription = document.getElementById('photoDescription');
    const scrollToTopButton = document.getElementById('scrollToTop');
    const relatedSection = document.getElementById('relatedSection');
    const relatedResults = document.getElementById('relatedResults');
    const relatedStatus = document.getElementById('relatedStatus');
    const relatedQueryText = document.getElementById('relatedQueryText');
    const similarityMix = document.getElementById('similarityMix');
    const similarityMixLabel = document.getElementById('similarityMixLabel');

    const pageSize = 30;
    const gridColumnCount = 4;
    const masonryColumns = [];
    const masonryHeights = [];
    const relatedColumns = [];
    const relatedHeights = [];

    let photos = [];
    let selectedIndex = 0;
    let recognition = null;
    let activeQuery = '';
    let currentPage = 0;
    let totalMatches = 0;
    let hasMore = false;
    let loadingMore = false;
    let searchToken = 0;
    let currentServerGeneration = 0;
    let activeSearchController = null;

    let relatedPhotos = [];
    let relatedPage = 0;
    let relatedTotal = 0;
    let relatedHasMore = false;
    let relatedLoading = false;
    let relatedToken = 0;
    let relatedServerGeneration = 0;
    let relatedController = null;
    let relatedDebounceTimer = null;
    let lastMixValue = '50';

    let isUserSeeking = false;
    let hudHideTimer = null;
    let viewerControlsTimer = null;

    const clientId = getClientId('familyPhotoClientId');
    const relatedClientId = getClientId('familyPhotoRelatedClientId');

    // Register Tizen TV Remote multimedia keys if available
    try {
        if (window.tizen && window.tizen.tvinputdevice) {
            ['MediaPlay', 'MediaPause', 'MediaPlayPause', 'MediaFastForward', 'MediaRewind', 'MediaStop'].forEach((k) => {
                try { window.tizen.tvinputdevice.registerKey(k); } catch (_) {}
            });
        }
    } catch (_) {}

    function getClientId(storageKey) {
        try {
            let id = localStorage.getItem(storageKey);
            if (!id) {
                id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
                localStorage.setItem(storageKey, id);
            }
            return id;
        } catch (_) {
            return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        }
    }

    function nextServerGeneration() {
        try {
            const generation = Number(localStorage.getItem('familyPhotoSearchGeneration') || 0) + 1;
            localStorage.setItem('familyPhotoSearchGeneration', String(generation));
            return generation;
        } catch (_) {
            return searchToken;
        }
    }

    function resetMasonry() {
        masonryColumns.length = 0;
        masonryHeights.length = 0;
        results.replaceChildren();
        for (let index = 0; index < gridColumnCount; index += 1) {
            const column = document.createElement('div');
            column.className = 'masonry-column';
            results.appendChild(column);
            masonryColumns.push(column);
            masonryHeights.push(0);
        }
    }

    function resetRelatedMasonry() {
        relatedColumns.length = 0;
        relatedHeights.length = 0;
        relatedResults.replaceChildren();
        for (let index = 0; index < gridColumnCount; index += 1) {
            const column = document.createElement('div');
            column.className = 'related-column';
            relatedResults.appendChild(column);
            relatedColumns.push(column);
            relatedHeights.push(0);
        }
    }

    const baseUrl = () => 'http://192.168.1.4:5000';
    const imageUrl = (route, filename) => `${baseUrl()}${route}/${filename.split('/').map(encodeURIComponent).join('/')}`;

    function visibleControls() {
        return [...document.querySelectorAll('button:not(:disabled), input:not([type="hidden"])')]
            .filter((element) => {
                if (element.hidden || element.closest('[hidden]')) return false;
                if (!viewer.classList.contains('related-mode') && viewer.contains(element) && !viewerHero.contains(element)) return false;
                const rect = element.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0 && getComputedStyle(element).visibility !== 'hidden';
            });
    }

    function moveFocus(direction, controlList, allowFallback = true) {
        const controls = controlList || visibleControls();
        if (!controls.length) return false;
        const current = document.activeElement;
        if (!controls.includes(current)) {
            controls[0].focus();
            return true;
        }

        const currentRect = current.getBoundingClientRect();
        const origin = { x: currentRect.left + currentRect.width / 2, y: currentRect.top + currentRect.height / 2 };
        const candidates = controls.filter((element) => element !== current).map((element) => {
            const rect = element.getBoundingClientRect();
            const point = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
            const dx = point.x - origin.x;
            const dy = point.y - origin.y;
            const forward = direction === 'ArrowRight' ? dx : direction === 'ArrowLeft' ? -dx : direction === 'ArrowDown' ? dy : -dy;
            const across = direction === 'ArrowLeft' || direction === 'ArrowRight' ? Math.abs(dy) : Math.abs(dx);
            return { element, forward, across, distance: Math.hypot(dx, dy) };
        });
        const ahead = candidates.filter((item) => item.forward > 2);
        if (!ahead.length && !allowFallback) return false;
        const pool = ahead.length ? ahead : candidates;
        pool.sort((a, b) => {
            const scoreA = ahead.length ? a.forward + a.across * 1.8 : a.distance + a.across * 2;
            const scoreB = ahead.length ? b.forward + b.across * 1.8 : b.distance + b.across * 2;
            return scoreA - scoreB;
        });
        pool[0]?.element.focus();
        return Boolean(pool[0]);
    }

    function centerFocusedPhoto(photo, instant = false) {
        if (!photo || !photo.classList.contains('photo')) return;
        const photoRect = photo.getBoundingClientRect();
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop || 0;
        const target = currentScroll + photoRect.top
            - Math.max(0, (window.innerHeight - photoRect.height) / 2);
        const pageHeight = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
        const maximum = Math.max(0, pageHeight - window.innerHeight);
        const finalScroll = Math.max(0, Math.min(maximum, target));
        if (instant) {
            const root = document.documentElement;
            const prev = root.style.scrollBehavior;
            root.style.scrollBehavior = 'auto';
            window.scrollTo(0, finalScroll);
            window.setTimeout(() => {
                root.style.scrollBehavior = prev;
            }, 60);
        } else {
            window.scrollTo(0, finalScroll);
        }
    }

    function isInRightmostColumn(photo) {
        const gridRect = results.getBoundingClientRect();
        const photoRect = photo.getBoundingClientRect();
        return photoRect.right >= gridRect.right - 12;
    }

    async function connect() {
        if (!baseUrl()) {
            status.textContent = 'Enter the laptop photo server address, for example http://192.168.1.4:5000.';
            status.classList.remove('connected');
            status.classList.add('connection-error');
            return;
        }
        status.textContent = 'Connecting to photo server…';
        status.classList.remove('connected', 'connection-error');
        try {
            const response = await fetch(`${baseUrl()}/`, { cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            status.textContent = 'Connected. Describe the photos you want to find.';
            status.classList.add('connected');
        } catch (error) {
            status.textContent = `Could not reach 192.168.1.4:5000 (${error.message}). Make sure the photo server is running.`;
            status.classList.add('connection-error');
        }
    }

    async function search(event) {
        if (event) event.preventDefault();
        searchToken += 1;
        const token = searchToken;
        if (activeSearchController) activeSearchController.abort();
        activeSearchController = typeof AbortController === 'function' ? new AbortController() : null;
        const controller = activeSearchController;
        const serverGeneration = nextServerGeneration();
        currentServerGeneration = serverGeneration;
        const text = queryInput.value.trim();
        if (!text) {
            activeQuery = '';
            photos = [];
            resetMasonry();
            loadingMore = false;
            status.textContent = 'Type a description or choose Voice Search.';
            queryInput.focus();
            return;
        }
        activeQuery = text;
        currentPage = 0;
        totalMatches = 0;
        hasMore = true;
        loadingMore = false;
        photos = [];
        resetMasonry();
        window.scrollTo(0, 0);
        scrollToTopButton.hidden = true;
        status.textContent = 'Searching your photo library…';
        status.classList.remove('connection-error');
        await loadNextPage(token, true, serverGeneration, controller);
    }

    async function loadNextPage(token, replace, serverGeneration, controller) {
        if (loadingMore || (!replace && !hasMore)) return;
        controller = controller || activeSearchController;
        loadingMore = true;
        const page = replace ? 1 : currentPage + 1;
        const form = new FormData();
        form.append('query_text', activeQuery);
        form.append('page', String(page));
        form.append('per_page', String(pageSize));
        form.append('client_id', clientId);
        form.append('search_generation', String(serverGeneration || currentServerGeneration || token));
        try {
            const response = await fetch(`${baseUrl()}/similar-images`, {
                method: 'POST', body: form, signal: controller?.signal
            });
            const data = await response.json();
            if (response.status === 409 && data.error === 'Search superseded') return;
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            if (token !== searchToken) return;
            const pagePhotos = data.results || (data.similar_images || []).map((path) => ({ type: 'image', path }));
            const firstIndex = photos.length;
            photos = replace ? pagePhotos : photos.concat(pagePhotos);
            currentPage = data.page || page;
            totalMatches = data.total_count || 0;
            hasMore = currentPage * pageSize < totalMatches;
            appendPhotos(pagePhotos, firstIndex, data.image_metadata || {});
            status.textContent = photos.length
                ? `${photos.length.toLocaleString()} of ${totalMatches.toLocaleString()} matches for “${activeQuery}”${hasMore ? ' · keep going down for more' : ''}`
                : `No matches for “${activeQuery}”. Try another description.`;
            if (replace && photos.length) results.querySelector('.photo')?.focus();
        } catch (error) {
            if (error.name === 'AbortError') return;
            if (token === searchToken) {
                status.textContent = `Could not load ${replace ? 'search results' : 'the next page'}: ${error.message}.`;
                status.classList.add('connection-error');
                hasMore = false;
            }
        } finally {
            if (token === searchToken) loadingMore = false;
        }
    }

    function appendPhotos(items, firstIndex, dimensions) {
        if (!masonryColumns.length) resetMasonry();
        const styles = getComputedStyle(results);
        const gap = parseFloat(styles.columnGap) || 0;
        const columnWidth = Math.max(1, (results.clientWidth - gap * (gridColumnCount - 1)) / gridColumnCount);
        items.forEach((item, offset) => {
            const path = typeof item === 'string' ? item : item.path;
            const isVideo = typeof item !== 'string' && item.type === 'video';
            const index = firstIndex + offset;
            const columnIndex = masonryHeights.indexOf(Math.min(...masonryHeights));
            const button = document.createElement('button');
            button.className = 'photo';
            button.type = 'button';
            button.dataset.photoIndex = String(index);
            button.setAttribute('aria-label', `Open ${isVideo ? 'video' : 'photo'} ${index + 1}`);
            button.addEventListener('click', () => openViewer(index));
            button.addEventListener('focus', () => {
                centerFocusedPhoto(button);
                const focusedIndex = Number(button.dataset.photoIndex);
                if (hasMore && focusedIndex >= photos.length - gridColumnCount * 2) {
                    loadNextPage(searchToken, false);
                }
            });
            const frame = document.createElement('span');
            frame.className = 'photo-frame';
            const size = dimensions[path];
            const naturalRatio = size && size.width > 0 && size.height > 0 ? size.height / size.width : null;
            const displayRatio = naturalRatio || 0.75;
            frame.style.paddingBottom = `${displayRatio * 100}%`;
            frame.classList.add('measured');
            const image = document.createElement('img');
            image.alt = `${isVideo ? 'Video' : 'Photo'} result ${index + 1}`;
            image.loading = 'eager';
            if (size && size.width > 0 && size.height > 0) {
                image.setAttribute('width', String(size.width));
                image.setAttribute('height', String(size.height));
            } else {
                image.addEventListener('load', () => {
                    if (image.naturalWidth && image.naturalHeight) {
                        const fallbackRatio = image.naturalHeight / image.naturalWidth;
                        frame.style.paddingBottom = `${fallbackRatio * 100}%`;
                        masonryHeights[columnIndex] += columnWidth * (fallbackRatio - displayRatio);
                    }
                }, { once: true });
            }
            image.src = isVideo ? `${baseUrl()}${item.poster_url}` : imageUrl('/images', path);
            frame.appendChild(image);
            if (isVideo) {
                const badge = document.createElement('span');
                badge.className = 'video-badge';
                badge.textContent = '▶';
                frame.appendChild(badge);
            }
            button.appendChild(frame);
            masonryColumns[columnIndex].appendChild(button);
            masonryHeights[columnIndex] += columnWidth * displayRatio + gap + 4;
        });
    }

    function updateSimilarityMixLabel() {
        const photoPercent = Number(similarityMix.value);
        similarityMixLabel.textContent = `Text ${100 - photoPercent}% · Photo ${photoPercent}%`;
    }

    function appendRelatedPhotos(items, firstIndex, dimensions) {
        if (!relatedColumns.length) resetRelatedMasonry();
        const styles = getComputedStyle(relatedResults);
        const gap = parseFloat(styles.columnGap) || 0;
        const columnWidth = Math.max(1, (relatedResults.clientWidth - gap * (gridColumnCount - 1)) / gridColumnCount);
        items.forEach((item, offset) => {
            const path = typeof item === 'string' ? item : item.path;
            const isVideo = typeof item !== 'string' && item.type === 'video';
            const index = firstIndex + offset;
            const columnIndex = relatedHeights.indexOf(Math.min(...relatedHeights));

            const button = document.createElement('button');
            button.className = 'related-photo';
            button.type = 'button';
            button.dataset.relatedIndex = String(index);
            button.setAttribute('aria-label', `Similar ${isVideo ? 'video' : 'photo'} ${index + 1}`);
            button.addEventListener('focus', () => centerRelatedPhoto(button));
            button.addEventListener('click', () => openRelatedItem(item));

            const frame = document.createElement('span');
            frame.className = 'related-photo-frame';
            const size = dimensions[path];
            const naturalRatio = size && size.width > 0 && size.height > 0 ? size.height / size.width : null;
            const displayRatio = naturalRatio || 0.75;
            frame.style.paddingBottom = `${displayRatio * 100}%`;

            const image = document.createElement('img');
            image.alt = '';
            image.loading = 'eager';
            if (size && size.width > 0 && size.height > 0) {
                image.setAttribute('width', String(size.width));
                image.setAttribute('height', String(size.height));
            } else {
                image.addEventListener('load', () => {
                    if (image.naturalWidth && image.naturalHeight) {
                        const fallbackRatio = image.naturalHeight / image.naturalWidth;
                        frame.style.paddingBottom = `${fallbackRatio * 100}%`;
                        relatedHeights[columnIndex] += columnWidth * (fallbackRatio - displayRatio);
                    }
                }, { once: true });
            }
            image.src = isVideo ? `${baseUrl()}${item.poster_url}` : imageUrl('/images', path);
            frame.appendChild(image);
            if (isVideo) {
                const badge = document.createElement('span');
                badge.className = 'video-badge';
                badge.textContent = '▶';
                frame.appendChild(badge);
            }
            button.appendChild(frame);
            relatedColumns[columnIndex].appendChild(button);
            relatedHeights[columnIndex] += columnWidth * displayRatio + gap + 4;
        });
    }

    function openRelatedItem(item) {
        exitRelatedMode();
        // Check if item is already in photos list
        const existingIdx = photos.findIndex((p) => {
            const pPath = typeof p === 'string' ? p : p.path;
            const iPath = typeof item === 'string' ? item : item.path;
            return pPath === iPath;
        });
        if (existingIdx >= 0) {
            selectedIndex = existingIdx;
        } else {
            photos.splice(selectedIndex + 1, 0, item);
            selectedIndex += 1;
        }
        showCurrentPhoto();
        viewerFeed.scrollTop = 0;
    }

    function centerRelatedPhoto(photo) {
        const feedRect = viewerFeed.getBoundingClientRect();
        const photoRect = photo.getBoundingClientRect();
        const target = viewerFeed.scrollTop + photoRect.top - feedRect.top
            - Math.max(0, (viewerFeed.clientHeight - photoRect.height) / 2);
        const maximum = Math.max(0, viewerFeed.scrollHeight - viewerFeed.clientHeight);
        viewerFeed.scrollTop = Math.max(0, Math.min(maximum, target));
    }

    async function loadRelatedPage(replace, advanceFocus = false, previousFocus = null) {
        if (!replace && (relatedLoading || !relatedHasMore)) return;
        if (replace) {
            relatedToken += 1;
            if (relatedController) relatedController.abort();
            relatedController = typeof AbortController === 'function' ? new AbortController() : null;
            relatedServerGeneration = nextServerGeneration();
            relatedPage = 0;
            relatedTotal = 0;
            relatedHasMore = true;
            relatedPhotos = [];
            relatedLoading = false;
            resetRelatedMasonry();
        }
        const token = relatedToken;
        const controller = relatedController;
        const page = replace ? 1 : relatedPage + 1;
        const form = new FormData();
        form.append('query_text', activeQuery);

        const selected = typeof photos[selectedIndex] === 'string'
            ? { type: 'image', path: photos[selectedIndex] }
            : (photos[selectedIndex] || {});
        if (selected.path) {
            form.append('query_image_filename', selected.path);
        }

        form.append('weight', String(Number(similarityMix.value)));
        form.append('page', String(page));
        form.append('per_page', String(pageSize));
        form.append('client_id', relatedClientId);
        form.append('search_generation', String(relatedServerGeneration));
        relatedLoading = true;
        relatedStatus.textContent = replace ? 'Finding similar moments…' : 'Loading more similar photos…';

        try {
            const response = await fetch(`${baseUrl()}/similar-images`, {
                method: 'POST', body: form, signal: controller?.signal
            });
            const data = await response.json();
            if (response.status === 409 && data.error === 'Search superseded') return;
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            if (token !== relatedToken) return;

            const pagePhotos = data.results || (data.similar_images || []).map((path) => ({ type: 'image', path }));
            const firstIndex = relatedPhotos.length;
            relatedPhotos = replace ? pagePhotos : relatedPhotos.concat(pagePhotos);
            relatedPage = data.page || page;
            relatedTotal = data.total_count || 0;
            relatedHasMore = relatedPage * pageSize < relatedTotal;
            appendRelatedPhotos(pagePhotos, firstIndex, data.image_metadata || {});
            relatedStatus.textContent = relatedPhotos.length
                ? `${relatedPhotos.length.toLocaleString()} of ${relatedTotal.toLocaleString()} similar moments${relatedHasMore ? ' · keep going down for more' : ''}`
                : 'No similar moments found. Try adjusting the Text ↔ Photo slider.';

            if (advanceFocus && previousFocus && token === relatedToken) {
                moveFocus('ArrowDown', [...relatedResults.querySelectorAll('.related-photo')], false);
            } else if (replace && pagePhotos.length > 0 && viewer.classList.contains('related-mode')) {
                // Focus first related tile if user is ready to browse results
                window.setTimeout(() => {
                    const firstTile = relatedResults.querySelector('.related-photo');
                    if (firstTile && document.activeElement !== similarityMix) {
                        firstTile.focus({ preventScroll: true });
                        centerRelatedPhoto(firstTile);
                    }
                }, 120);
            }
        } catch (error) {
            if (error.name === 'AbortError') return;
            if (token === relatedToken) {
                relatedStatus.textContent = `Could not load similar moments: ${error.message}.`;
                relatedHasMore = false;
            }
        } finally {
            if (token === relatedToken) relatedLoading = false;
        }
    }

    function maybeLoadRelatedMore() {
        if (!viewer.classList.contains('related-mode') || relatedLoading || !relatedHasMore) return;
        const remaining = viewerFeed.scrollHeight - viewerFeed.scrollTop - viewerFeed.clientHeight;
        if (remaining < Math.max(220, viewerFeed.clientHeight * 0.35)) loadRelatedPage(false);
    }

    function focusMainPhoto() {
        viewerHero.scrollIntoView({ behavior: 'smooth' });
        if (!largeVideo.hidden) {
            videoProgressBar.focus();
        } else {
            largePhoto.focus();
        }
    }

    function enterRelatedMode() {
        if (viewer.classList.contains('related-mode')) return;
        viewer.classList.add('related-mode');
        relatedQueryText.textContent = activeQuery ? `Text context: “${activeQuery}”` : 'Searching visually from this moment';
        similarityMix.value = '50';
        lastMixValue = '50';
        updateSimilarityMixLabel();
        relatedStatus.textContent = 'Finding similar moments…';
        relatedSection.scrollIntoView({ behavior: 'smooth' });
        loadRelatedPage(true);
    }

    function exitRelatedMode() {
        const wasRelated = viewer.classList.contains('related-mode');
        if (relatedDebounceTimer) window.clearTimeout(relatedDebounceTimer);
        if (relatedController) relatedController.abort();
        relatedToken += 1;
        relatedLoading = false;
        viewer.classList.remove('related-mode');
        viewerHero.scrollIntoView({ behavior: 'smooth' });
        if (wasRelated) {
            const form = new FormData();
            form.append('client_id', relatedClientId);
            form.append('search_generation', String(nextServerGeneration()));
            fetch(`${baseUrl()}/search-cancel`, { method: 'POST', body: form }).catch(() => {});
        }
    }

    function maybeLoadMore() {
        if (!hasMore || loadingMore || !activeQuery) return;
        const pageHeight = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop || 0;
        const remaining = pageHeight - currentScroll - window.innerHeight;
        if (remaining < Math.max(180, window.innerHeight * 0.3)) {
            loadNextPage(searchToken, false);
        }
    }

    function onResultsScroll() {
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop || 0;
        scrollToTopButton.hidden = currentScroll < 280;
        maybeLoadMore();
    }

    function scrollResultsToTop() {
        window.scrollTo(0, 0);
        scrollToTopButton.hidden = true;
        window.setTimeout(() => results.querySelector('.photo')?.focus(), 0);
    }

    // Video Player & HUD Functions
    function toggleVideoPlayPause() {
        if (largeVideo.hidden) return;
        if (largeVideo.paused || largeVideo.ended) {
            const playPromise = largeVideo.play();
            if (playPromise && playPromise.then) {
                playPromise.then(() => {
                    updatePlayPauseButton(true);
                    showVideoHud('▶');
                }).catch(() => {});
            } else {
                updatePlayPauseButton(true);
                showVideoHud('▶');
            }
        } else {
            largeVideo.pause();
            updatePlayPauseButton(false);
            showVideoHud('❚❚');
        }
        revealViewerControls();
    }

    function updatePlayPauseButton(isPlaying) {
        if (!videoPlayPause) return;
        videoPlayPause.textContent = isPlaying ? '❚❚' : '▶';
        videoPlayPause.setAttribute('aria-label', isPlaying ? 'Pause video' : 'Play video');
    }

    function showVideoHud(text) {
        if (!videoFeedbackHud) return;
        videoFeedbackHud.textContent = text;
        videoFeedbackHud.hidden = false;
        videoFeedbackHud.classList.remove('hud-pulse');
        void videoFeedbackHud.offsetWidth; // Trigger reflow for animation restart
        videoFeedbackHud.classList.add('hud-pulse');
        if (hudHideTimer) window.clearTimeout(hudHideTimer);
        hudHideTimer = window.setTimeout(() => {
            videoFeedbackHud.hidden = true;
        }, 850);
    }

    function seekVideoRelative(seconds) {
        if (largeVideo.hidden || !largeVideo.duration) return;
        const target = Math.max(0, Math.min(largeVideo.duration, (largeVideo.currentTime || 0) + seconds));
        largeVideo.currentTime = target;
        updateVideoProgressDisplay();
        showVideoHud(seconds > 0 ? `+${seconds}s` : `${seconds}s`);
        revealViewerControls();
    }

    function seekVideoToPercent(percent) {
        if (largeVideo.hidden || !largeVideo.duration) return;
        const target = Math.max(0, Math.min(largeVideo.duration, (percent / 100) * largeVideo.duration));
        largeVideo.currentTime = target;
        updateVideoProgressDisplay();
        revealViewerControls();
    }

    function updateVideoProgressDisplay() {
        if (largeVideo.hidden) return;
        const dur = largeVideo.duration || 0;
        const cur = largeVideo.currentTime || 0;
        videoCurrentTime.textContent = formatDuration(cur);
        videoDuration.textContent = formatDuration(dur);
        if (!isUserSeeking && dur > 0) {
            const pct = (cur / dur) * 100;
            videoProgressBar.value = pct.toFixed(1);
            videoProgressBar.style.setProperty('--progress-pct', `${pct}%`);
        }
    }

    function openViewer(index) {
        if (!photos.length) return;
        exitRelatedMode();
        selectedIndex = (index + photos.length) % photos.length;
        viewerFeed.scrollTop = 0;
        viewer.classList.remove('hidden');
        showCurrentPhoto();
        revealViewerControls();
    }

    function showCurrentPhoto() {
        const item = typeof photos[selectedIndex] === 'string'
            ? { type: 'image', path: photos[selectedIndex] } : photos[selectedIndex];
        const path = item.path;
        const name = item.name || path.split('/').pop();

        if (item.type === 'video') {
            largePhoto.hidden = true;
            largePhoto.removeAttribute('src');
            largeVideo.hidden = false;
            videoControls.hidden = false;
            viewerHelp.textContent = 'Center: Play/Pause　　Left/Right: Seek 10s　　Down: Similar moments　　Back: Close';

            largeVideo.src = `${baseUrl()}${item.video_url}`;
            largeVideo.load();
            videoProgressBar.value = '0';
            videoProgressBar.style.setProperty('--progress-pct', '0%');
            videoCurrentTime.textContent = '0:00';
            videoDuration.textContent = formatDuration(item.duration_seconds || 0);
            updatePlayPauseButton(false);

            const playback = largeVideo.play();
            if (playback && playback.then) {
                playback.then(() => updatePlayPauseButton(true)).catch(() => updatePlayPauseButton(false));
            }
            photoDescription.textContent = `Video · sampled match at ${formatDuration(item.match_timestamp)}${activeQuery ? ` · Search: ${activeQuery}` : ''}`;
            window.setTimeout(() => videoProgressBar.focus(), 120);
        } else {
            largeVideo.pause();
            largeVideo.removeAttribute('src');
            largeVideo.load();
            largeVideo.hidden = true;
            videoControls.hidden = true;
            viewerHelp.textContent = '← Previous　　Next →　　Down: Similar moments　　Back: Close';

            largePhoto.hidden = false;
            largePhoto.removeAttribute('src');
            largePhoto.src = imageUrl('/displayimages', path);
            photoDescription.textContent = activeQuery ? `Search: ${activeQuery}` : 'Family photo';
            window.setTimeout(() => largePhoto.focus(), 120);
        }

        photoCaption.textContent = `${selectedIndex + 1} of ${photos.length} · ${item.type === 'video' ? '▶ ' : ''}${name}`;
        photoName.textContent = name;
        revealViewerControls();
    }

    function formatDuration(seconds) {
        const value = Math.max(0, Number(seconds) || 0);
        const minutes = Math.floor(value / 60);
        const remainder = Math.floor(value % 60).toString().padStart(2, '0');
        return `${minutes}:${remainder}`;
    }

    function revealViewerControls() {
        if (viewer.classList.contains('hidden')) return;
        viewer.classList.remove('ui-hidden');
        if (viewerControlsTimer) window.clearTimeout(viewerControlsTimer);
        viewerControlsTimer = window.setTimeout(() => {
            // Keep controls visible if user is actively focused on video controls or slider
            if (document.activeElement && videoControls && videoControls.contains(document.activeElement)) return;
            viewer.classList.add('ui-hidden');
        }, 4000);
    }

    function closeViewer() {
        if (viewerControlsTimer) window.clearTimeout(viewerControlsTimer);
        exitRelatedMode();
        largeVideo.pause();
        largeVideo.removeAttribute('src');
        largeVideo.load();
        largeVideo.hidden = true;
        videoControls.hidden = true;
        largePhoto.hidden = false;
        viewer.classList.add('hidden');
        viewer.classList.remove('ui-hidden');

        const targetPhoto = results.querySelector(`.photo[data-photo-index="${selectedIndex}"]`);
        if (targetPhoto) {
            targetPhoto.focus({ preventScroll: true });
            centerFocusedPhoto(targetPhoto, true);
            window.requestAnimationFrame(() => {
                centerFocusedPhoto(targetPhoto, true);
            });
        }
    }

    function movePhoto(delta) {
        if (viewer.classList.contains('hidden') || !photos.length) return;
        if (viewer.classList.contains('related-mode')) exitRelatedMode();
        selectedIndex = (selectedIndex + delta + photos.length) % photos.length;
        showCurrentPhoto();
        if (hasMore && selectedIndex >= photos.length - 8) {
            loadNextPage(searchToken, false);
        }
    }

    function startVoiceSearch() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            status.textContent = 'Voice recognition is not available in this TV browser. Type a search with the remote keyboard.';
            queryInput.focus();
            return;
        }
        if (recognition) recognition.abort();
        recognition = new SpeechRecognition();
        recognition.lang = navigator.language || 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        let completedTranscript = '';
        status.textContent = 'Listening… speak your photo search now.';
        recognition.onresult = (event) => {
            const finalParts = [];
            for (let i = 0; i < event.results.length; i += 1) {
                if (event.results[i].isFinal) finalParts.push(event.results[i][0].transcript);
            }
            completedTranscript = finalParts.join(' ').trim();
            if (completedTranscript) queryInput.value = completedTranscript;
        };
        recognition.onerror = (event) => {
            status.textContent = event.error === 'not-allowed'
                ? 'Microphone permission was denied. Allow microphone access for this app, then try again.'
                : `Voice search ended (${event.error}). You can type your search instead.`;
        };
        recognition.onend = () => {
            if (completedTranscript) {
                queryInput.value = completedTranscript;
                search();
            } else if (status.textContent.startsWith('Listening')) {
                status.textContent = 'No speech heard. Select Voice and try again.';
            }
        };
        recognition.start();
    }

    // Video Element Listeners
    largeVideo.addEventListener('timeupdate', () => {
        if (!isUserSeeking) updateVideoProgressDisplay();
    });
    largeVideo.addEventListener('loadedmetadata', updateVideoProgressDisplay);
    largeVideo.addEventListener('durationchange', updateVideoProgressDisplay);
    largeVideo.addEventListener('play', () => updatePlayPauseButton(true));
    largeVideo.addEventListener('pause', () => updatePlayPauseButton(false));
    largeVideo.addEventListener('ended', () => {
        updatePlayPauseButton(false);
        showVideoHud('↺');
    });

    videoPlayPause.addEventListener('click', toggleVideoPlayPause);
    videoRewind.addEventListener('click', () => seekVideoRelative(-10));
    videoForward.addEventListener('click', () => seekVideoRelative(10));

    videoProgressBar.addEventListener('input', () => {
        isUserSeeking = true;
        const pct = parseFloat(videoProgressBar.value);
        videoProgressBar.style.setProperty('--progress-pct', `${pct}%`);
        if (largeVideo.duration) {
            const time = (pct / 100) * largeVideo.duration;
            videoCurrentTime.textContent = formatDuration(time);
        }
    });

    videoProgressBar.addEventListener('change', () => {
        const pct = parseFloat(videoProgressBar.value);
        seekVideoToPercent(pct);
        isUserSeeking = false;
    });

    // App & Viewer Controls Listeners
    document.getElementById('searchForm').addEventListener('submit', search);
    document.getElementById('voiceButton').addEventListener('click', startVoiceSearch);
    window.addEventListener('scroll', onResultsScroll);
    scrollToTopButton.addEventListener('click', scrollResultsToTop);
    document.getElementById('previousPhoto').addEventListener('click', () => movePhoto(-1));
    document.getElementById('nextPhoto').addEventListener('click', () => movePhoto(1));
    document.getElementById('closeViewer').addEventListener('click', closeViewer);
    viewerFeed.addEventListener('scroll', maybeLoadRelatedMore);

    similarityMix.addEventListener('input', () => {
        if (similarityMix.value === lastMixValue) return;
        lastMixValue = similarityMix.value;
        updateSimilarityMixLabel();
        if (!viewer.classList.contains('related-mode')) return;
        if (relatedDebounceTimer) window.clearTimeout(relatedDebounceTimer);
        relatedDebounceTimer = window.setTimeout(() => loadRelatedPage(true), 400);
    });

    // Keydown remote navigation
    document.addEventListener('keydown', (event) => {
        const keyMap = {
            37: 'ArrowLeft', 38: 'ArrowUp', 39: 'ArrowRight', 40: 'ArrowDown',
            13: 'Enter', 10009: 'BrowserBack', 27: 'Escape', 8: 'Backspace',
            415: 'MediaPlay', 19: 'MediaPause', 10252: 'MediaPlayPause',
            417: 'MediaFastForward', 412: 'MediaRewind', 413: 'MediaStop'
        };
        const key = keyMap[event.keyCode] || event.key;
        const inViewer = !viewer.classList.contains('hidden');
        const isVideoActive = inViewer && !largeVideo.hidden;
        const editingText = document.activeElement instanceof HTMLInputElement && document.activeElement.type === 'text';
        const active = document.activeElement;

        if (inViewer) revealViewerControls();

        // Hardware Media Keys
        if (isVideoActive) {
            if (key === 'MediaPlay') { event.preventDefault(); largeVideo.play(); showVideoHud('▶'); return; }
            if (key === 'MediaPause') { event.preventDefault(); largeVideo.pause(); showVideoHud('❚❚'); return; }
            if (key === 'MediaPlayPause') { event.preventDefault(); toggleVideoPlayPause(); return; }
            if (key === 'MediaFastForward') { event.preventDefault(); seekVideoRelative(15); return; }
            if (key === 'MediaRewind') { event.preventDefault(); seekVideoRelative(-15); return; }
        }

        // Remote Center Button (Enter / Select)
        if (key === 'Enter' || key === 'Select') {
            if (editingText) return; // Allow form submit

            if (isVideoActive && (active === videoProgressBar || active === largeVideo || active === videoPlayPause || active === viewerHero)) {
                event.preventDefault();
                toggleVideoPlayPause();
                return;
            }

            if (active instanceof HTMLButtonElement) {
                event.preventDefault();
                active.click();
                return;
            }
        }

        // In Related Mode navigation
        if (inViewer && viewer.classList.contains('related-mode')) {
            const isRelatedTile = active?.classList?.contains('related-photo');
            const onHero = viewerHero.contains(active);

            if ((key === 'ArrowLeft' || key === 'ArrowRight') && onHero && !isVideoActive) {
                event.preventDefault();
                movePhoto(key === 'ArrowRight' ? 1 : -1);
                return;
            }

            if (active === similarityMix && (key === 'ArrowLeft' || key === 'ArrowRight')) {
                // Allow slider change natively
                return;
            }

            if (key === 'ArrowUp' && active === similarityMix) {
                event.preventDefault();
                focusMainPhoto();
                return;
            }

            if (key === 'ArrowDown' && onHero) {
                event.preventDefault();
                similarityMix.focus();
                return;
            }

            if (key === 'ArrowDown' && active === similarityMix) {
                event.preventDefault();
                relatedResults.querySelector('.related-photo')?.focus();
                return;
            }

            if (isRelatedTile && (key === 'ArrowUp' || key === 'ArrowDown' || key === 'ArrowLeft' || key === 'ArrowRight')) {
                event.preventDefault();
                if (key === 'ArrowUp') {
                    // Check if in top row of related items
                    const allTiles = [...relatedResults.querySelectorAll('.related-photo')];
                    const idx = allTiles.indexOf(active);
                    if (idx < gridColumnCount) {
                        similarityMix.focus();
                        return;
                    }
                }
                const moved = moveFocus(key, [...relatedResults.querySelectorAll('.related-photo')], false);
                if (!moved && key === 'ArrowDown' && relatedHasMore) {
                    loadRelatedPage(false, true, active);
                }
                return;
            }
        } else if (inViewer) {
            // In Viewer Hero (Standard View)
            if (isVideoActive) {
                // Video is active: Left/Right seeks on progress bar, video, or hero!
                if (key === 'ArrowLeft' || key === 'ArrowRight') {
                    if (active === videoProgressBar || active === largeVideo || active === viewerHero) {
                        event.preventDefault();
                        seekVideoRelative(key === 'ArrowRight' ? 10 : -10);
                        return;
                    }
                }
                if (key === 'ArrowDown') {
                    event.preventDefault();
                    enterRelatedMode();
                    return;
                }
                if (key === 'ArrowUp' && (active === videoProgressBar || active === videoPlayPause || active === videoRewind || active === videoForward)) {
                    event.preventDefault();
                    document.getElementById('closeViewer')?.focus();
                    return;
                }
            } else {
                // Photo is active
                if (key === 'ArrowDown') {
                    event.preventDefault();
                    enterRelatedMode();
                    return;
                }
                if (key === 'ArrowRight') { event.preventDefault(); movePhoto(1); return; }
                if (key === 'ArrowLeft') { event.preventDefault(); movePhoto(-1); return; }
            }
        }

        // Gallery navigation (outside viewer)
        if (!inViewer && key === 'ArrowRight' && active.classList?.contains('photo') && isInRightmostColumn(active)) {
            event.preventDefault();
            if (!scrollToTopButton.hidden) scrollToTopButton.focus();
            return;
        }

        // Generic arrow navigation
        if (key === 'ArrowUp' || key === 'ArrowDown' || ((!editingText || inViewer) && (key === 'ArrowLeft' || key === 'ArrowRight'))) {
            if (!inViewer && key === 'ArrowDown' && active.classList?.contains('photo')) {
                const focusedIndex = Number(active.dataset.photoIndex);
                if (hasMore && focusedIndex >= photos.length - 8) loadNextPage(searchToken, false);
            }
            event.preventDefault();
            moveFocus(key);
            return;
        }

        // Back / Escape key
        if (key === 'Escape' || key === 'Backspace' || key === 'BrowserBack' || key === 'Return') {
            if (!inViewer) return;
            event.preventDefault();
            if (viewer.classList.contains('related-mode')) {
                exitRelatedMode();
                focusMainPhoto();
                return;
            }
            closeViewer();
        }
    });

    window.addEventListener('load', () => {
        queryInput.focus();
        connect();
    });
})();
