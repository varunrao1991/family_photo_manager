const config = {
    perPage: 30,
    infiniteScrollThreshold: 20,
    zoomStep: 0.2,
    panBorderSize: 50,
    panSensitivity: 10,
};

let state = {
    currentPage: 1,
    isLoading: false,
    hasMore: true,
    selectedImages: new Set(),
    lastSearch: null,
    zoomLevel: 1,
    currentImage: null,
    isDragging: false,
    startX: 0,
    startY: 0,
    translateX: 0,
    translateY: 0,
    savedTransform: "",
    panDirection: { x: 0, y: 0 },
    panInterval: null,
    pendingDeletion: new Set(), // Track images being deleted to prevent duplicate calls
};

const elements = {
    searchForm: document.getElementById("searchForm"),
    weightSlider: document.getElementById("weightScrollbar"),
    weightDisplay: document.getElementById("weightDisplay"),
    results: document.getElementById("results"),
    loading: document.getElementById("loading"),
    actionBar: document.getElementById("actionBar"),
    selectAll: document.getElementById("selectAll"),
    rotateLeftSelected: document.getElementById("rotateLeftSelected"),
    rotateRightSelected: document.getElementById("rotateRightSelected"),
    deleteSelected: document.getElementById("deleteSelected"),
    selectedCount: document.getElementById("selected-count"),
    queryText: document.getElementById("query_text"),
    queryImage: document.getElementById("query_image"),
    previewImage: document.getElementById("previewImage"),
    closeButton: document.getElementById("clearpreview"),
    infiniteScrollTrigger: document.getElementById("infiniteScrollTrigger"),
    imageModalElement: document.getElementById("imageModal"),
    largeImage: document.getElementById("largeImage"),
    zoomInModal: document.getElementById("zoomInModal"),
    zoomOutModal: document.getElementById("zoomOutModal"),
    rotateLeftModal: document.getElementById("rotateLeftModal"),
    rotateRightModal: document.getElementById("rotateRightModal"),
    downloadImage: document.getElementById("downloadImage"),
    helpModalElement: document.getElementById("helpModal"),
    toggleHelp: document.getElementById("toggle-help"),
    deleteConfirmationModalElement: document.getElementById("deleteConfirmationModal"),
    confirmationThumbnails: document.getElementById("confirmationThumbnails"),
    deleteCount: document.getElementById("delete-count"),
    confirmDeleteBtn: document.getElementById("confirmDeleteBtn"),
    notificationContainer: document.getElementById("notificationContainer"),
    imageModal: null,
    helpModal: null,
    deleteConfirmationModal: null,
};

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    initializeModals();
    setupEventListeners();
    setupIntersectionObserver();
});

function initializeModals() {
    if (elements.imageModalElement) {
        elements.imageModal = new bootstrap.Modal(elements.imageModalElement);
    }
    if (elements.helpModalElement) {
        elements.helpModal = new bootstrap.Modal(elements.helpModalElement);
    }
    if (elements.deleteConfirmationModalElement) {
        elements.deleteConfirmationModal = new bootstrap.Modal(
            elements.deleteConfirmationModalElement
        );
    }
}

// Event Listeners
function setupEventListeners() {
    // Image manipulation
    document.getElementById('resetPosition')?.addEventListener('click', resetImagePosition);
    elements.zoomInModal?.addEventListener('click', zoomInImage);
    elements.zoomOutModal?.addEventListener('click', zoomOutImage);
    elements.rotateLeftModal?.addEventListener('click', () => rotateImage("left"));
    elements.rotateRightModal?.addEventListener('click', () => rotateImage("right"));
    elements.downloadImage?.addEventListener('click', downloadImage);

    // Search and selection
    elements.searchForm?.addEventListener('submit', handleSearchSubmit);
    elements.queryImage?.addEventListener('change', handleFileSelect);
    elements.selectAll?.addEventListener('click', selectAllImages);
    elements.closeButton?.addEventListener('click', handleClearSearchImage);
    elements.weightSlider.addEventListener("input", (e) => {
        updateWeightLabel(e.target.value);
    });

    // Bulk actions
    elements.rotateLeftSelected?.addEventListener('click', () => rotateSelected('left'));
    elements.rotateRightSelected?.addEventListener('click', () => rotateSelected('right'));
    elements.deleteSelected?.addEventListener('click', showDeleteConfirmation);
    elements.confirmDeleteBtn?.addEventListener('click', handleConfirmedDelete);

    // Modal interactions
    elements.toggleHelp?.addEventListener('click', () => elements.helpModal?.show());

    // Window events
    window.addEventListener('scroll', handleScroll);

    // Image modal events
    if (elements.imageModalElement) {
        elements.imageModalElement.addEventListener('mousemove', handleModalMouseMove);
        elements.imageModalElement.addEventListener('mouseleave', stopEdgePan);
        elements.imageModalElement.addEventListener('hidden.bs.modal', resetModalState);
    }
}

// Image Viewing and Manipulation
function resetImagePosition() {
    if (!elements.largeImage) return;
    state.translateX = 0;
    state.translateY = 0;
    applyImageTransform();
}

function zoomInImage() {
    state.zoomLevel += config.zoomStep;
    applyImageTransform();
}

function zoomOutImage() {
    state.zoomLevel = Math.max(0.1, state.zoomLevel - config.zoomStep);
    applyImageTransform();
}

function applyImageTransform() {
    if (!elements.largeImage) return;
    elements.largeImage.style.transform = `
        translate(${state.translateX}px, ${state.translateY}px)
        scale(${state.zoomLevel})
    `;
}

function handleModalMouseMove(e) {
    if (!elements.largeImage || state.isDragging) return;

    const modal = elements.imageModalElement.querySelector('.modal-dialog');
    const modalRect = modal.getBoundingClientRect();
    const imgRect = elements.largeImage.getBoundingClientRect();

    const leftSpace = imgRect.left - modalRect.left;
    const rightSpace = modalRect.right - imgRect.right;
    const topSpace = imgRect.top - modalRect.top;
    const bottomSpace = modalRect.bottom - imgRect.bottom;

    const shouldPanLeft = leftSpace < 0 && e.clientX < modalRect.left + config.panBorderSize;
    const shouldPanRight = rightSpace < 0 && e.clientX > modalRect.right - config.panBorderSize;
    const shouldPanUp = topSpace < 0 && e.clientY < modalRect.top + config.panBorderSize;
    const shouldPanDown = bottomSpace < 0 && e.clientY > modalRect.bottom - config.panBorderSize;

    const panX = shouldPanLeft ? 1 : shouldPanRight ? -1 : 0;
    const panY = shouldPanUp ? 1 : shouldPanDown ? -1 : 0;

    if (panX !== state.panDirection.x || panY !== state.panDirection.y) {
        state.panDirection = { x: panX, y: panY };
        if (panX !== 0 || panY !== 0) {
            startEdgePan();
        } else {
            stopEdgePan();
        }
    }
}

function startEdgePan() {
    if (state.panInterval) return;

    state.panInterval = setInterval(() => {
        const moveX = state.panDirection.x * config.panSensitivity;
        const moveY = state.panDirection.y * config.panSensitivity;

        state.translateX += moveX;
        state.translateY += moveY;

        applyImageTransform();
    }, 16);
}

function stopEdgePan() {
    if (state.panInterval) {
        clearInterval(state.panInterval);
        state.panInterval = null;
    }
    state.panDirection = { x: 0, y: 0 };
}

function resetModalState() {
    stopEdgePan();
    state.zoomLevel = 1;
    state.translateX = 0;
    state.translateY = 0;
    if (elements.largeImage) {
        elements.largeImage.style.transform = '';
    }
}

// Search and Results Handling
function setupIntersectionObserver() {
    if (!elements.infiniteScrollTrigger) return;
    const observer = new IntersectionObserver(
        (entries) => {
            if (entries[0].isIntersecting && !state.isLoading && state.hasMore && state.lastSearch) {
                loadMoreResults();
            }
        },
        { threshold: 0.1 }
    );
    observer.observe(elements.infiniteScrollTrigger);
}

function updateWeightLabel(value) {
    const textPercent = 100 - value;
    elements.weightDisplay.textContent = `${textPercent}`;
}

// Initial load
updateWeightLabel(elements.weightSlider.value);

async function handleSearchSubmit(e) {
    e.preventDefault();
    resetSearchState();

    const formData = new FormData();
    if (elements.queryText.value) {
        formData.append("query_text", elements.queryText.value);
    }
    if (elements.queryImage.files[0]) {
        formData.append("query_image", elements.queryImage.files[0]);
    }
    formData.append("weight", elements.weightSlider.value);
    formData.append("page", state.currentPage);
    formData.append("per_page", config.perPage);

    state.lastSearch = formData;
    await performSearch(formData);
}

function resetSearchState() {
    state.currentPage = 1;
    state.hasMore = true;
    state.selectedImages.clear();
    updateActionBar();
    elements.results.innerHTML = "";
    toggleLoading(true);
}

async function performSearch(formData) {
    try {
        const response = await fetch("/similar-images", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        displayResults(data.similar_images || []);
        state.hasMore = data.similar_images ? data.similar_images.length === config.perPage : false;
    } catch (error) {
        console.error("Search error:", error);
        showNotification(`Search failed: ${error.message}`, "error");
    } finally {
        toggleLoading(false);
    }
}

async function loadMoreResults() {
    if (state.isLoading || !state.hasMore || !state.lastSearch) return;

    state.currentPage++;
    toggleLoading(true);

    try {
        const formData = state.lastSearch;
        formData.set("page", state.currentPage);

        const response = await fetch("/similar-images", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        if (data.similar_images && data.similar_images.length > 0) {
            displayResults(data.similar_images, true);
            state.hasMore = data.similar_images.length === config.perPage;
        } else {
            state.hasMore = false;
        }
    } catch (error) {
        console.error("Load more error:", error);
        showNotification(`Failed to load more images: ${error.message}`, "error");
    } finally {
        toggleLoading(false);
    }
}

function displayResults(images, append = false) {
    if (!append) {
        elements.results.innerHTML = "";
    }

    if (images.length === 0) {
        handleEmptyResults(append);
        return;
    }

    images.forEach((imagePath) => {
        createImageCard(imagePath);
    });
}

function handleEmptyResults(append) {
    if (!append && state.lastSearch) {
        elements.results.innerHTML = `
            <div class="col-12 text-center py-5">
                <i class="fas fa-image fa-3x mb-3 text-muted"></i>
                <h5 class="text-muted">No more images found for this search</h5>
            </div>
        `;
        state.hasMore = false;
    } else if (!append && !state.lastSearch) {
        elements.results.innerHTML = `
            <div class="col-12 text-center py-5">
                <i class="fas fa-image fa-3x mb-3 text-muted"></i>
                <h5 class="text-muted">No images found</h5>
                <p>Try a different search term or upload another image</p>
            </div>
        `;
    }
}

function createImageCard(imagePath) {
    const fileName = imagePath.split("/").pop();
    const card = document.createElement("div");
    card.className = "photo-card";
    card.dataset.imagePath = imagePath;

    if (state.selectedImages.has(imagePath)) {
        card.classList.add("selected");
    }

    const img = document.createElement("img");
    img.className = "photo-img";
    img.src = `/images/${imagePath}?t=${Date.now()}`;
    img.loading = "lazy";
    img.alt = fileName;

    const overlay = document.createElement("div");
    overlay.className = "photo-overlay";
    overlay.textContent = fileName;

    const actions = document.createElement("div");
    actions.className = "photo-actions";

    // Action buttons
    const buttons = [
        { icon: "fa-undo", title: "Rotate Left", action: () => rotateSingleImage(imagePath, "left") },
        { icon: "fa-redo", title: "Rotate Right", action: () => rotateSingleImage(imagePath, "right") },
        { icon: "fa-link", title: "Copy Path", action: () => copyImagePathToClipboard(imagePath) },
        { icon: "fa-trash", title: "Delete", action: () => showSingleDeleteConfirmation(imagePath) },
        { icon: "fa-download", title: "Download", action: () => downloadSingleImage(imagePath) }
    ];

    buttons.forEach(btn => {
        const button = document.createElement("button");
        button.className = "photo-action-btn";
        button.innerHTML = `<i class="fas ${btn.icon}"></i>`;
        button.title = btn.title;
        button.addEventListener("click", (e) => {
            e.stopPropagation();
            btn.action();
        });
        actions.appendChild(button);
    });

    card.addEventListener("click", () => toggleImageSelection(card));
    card.addEventListener("dblclick", (e) => {
        e.preventDefault();
        e.stopPropagation();
        openImageModal(imagePath, fileName);
    });

    card.appendChild(img);
    card.appendChild(overlay);
    card.appendChild(actions);
    elements.results.appendChild(card);
}

// Image Selection and Actions
function toggleImageSelection(card) {
    const imagePath = card.dataset.imagePath;
    if (!imagePath) return;

    if (state.selectedImages.has(imagePath)) {
        state.selectedImages.delete(imagePath);
        card.classList.remove("selected");
    } else {
        state.selectedImages.add(imagePath);
        card.classList.add("selected");
    }
    updateActionBar();
}

function selectAllImages() {
    const allCards = document.querySelectorAll(".photo-card");
    const allPaths = Array.from(allCards)
        .map((card) => card.dataset.imagePath)
        .filter((path) => path);

    if (state.selectedImages.size === allPaths.length && allPaths.length > 0) {
        // Deselect all
        allCards.forEach((card) => card.classList.remove("selected"));
        state.selectedImages.clear();
    } else {
        // Select all
        allCards.forEach((card) => {
            if (card.dataset.imagePath && !state.selectedImages.has(card.dataset.imagePath)) {
                card.classList.add("selected");
                state.selectedImages.add(card.dataset.imagePath);
            }
        });
    }
    updateActionBar();
}

function updateActionBar() {
    const count = state.selectedImages.size;

    if (count > 0) {
        elements.actionBar?.classList.add("visible");
        if (elements.selectedCount) {
            elements.selectedCount.classList.remove("d-none");
            elements.selectedCount.textContent = `${count} selected`;
        }

        // Enable action buttons
        if (elements.rotateLeftSelected) elements.rotateLeftSelected.disabled = false;
        if (elements.rotateRightSelected) elements.rotateRightSelected.disabled = false;
        if (elements.deleteSelected) elements.deleteSelected.disabled = false;
    } else {
        elements.actionBar?.classList.remove("visible");
        if (elements.selectedCount) {
            elements.selectedCount.classList.add("d-none");
        }

        // Disable action buttons
        if (elements.rotateLeftSelected) elements.rotateLeftSelected.disabled = true;
        if (elements.rotateRightSelected) elements.rotateRightSelected.disabled = true;
        if (elements.deleteSelected) elements.deleteSelected.disabled = true;
    }

    // Update select all button text
    if (elements.selectAll) {
        elements.selectAll.textContent =
            state.selectedImages.size === document.querySelectorAll(".photo-card").length &&
                document.querySelectorAll(".photo-card").length > 0
                ? "Deselect All"
                : "Select All";
    }
}

// Image Operations
async function rotateSingleImage(imagePath, direction) {
    const notification = showNotification(`Rotating image...`, "info", 0);
    try {
        const response = await fetch("/rotate-images", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ images: [imagePath], direction }),
        });

        const result = await response.json();
        if (result.success) {
            refreshImageInUI(imagePath);

            if (notification) {
                updateNotification(notification, "success", "Image rotated");
            }
        } else {
            throw new Error(result.error || "Rotation failed on server");
        }
    } catch (error) {
        console.error("Rotate error:", error);
        if (notification) {
            updateNotification(notification, "error", "Rotation failed");
        }
    }
}

async function rotateSelected(direction) {
    if (state.selectedImages.size === 0) return;

    const notification = showNotification(
        `Rotating ${state.selectedImages.size} images...`,
        "info",
        0
    );

    try {
        const response = await fetch("/rotate-images", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ images: Array.from(state.selectedImages), direction }),
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const result = await response.json();
        if (result.success) {
            // Refresh all selected images in the UI
            state.selectedImages.forEach(imagePath => {
                refreshImageInUI(imagePath);
            });

            // Clear selection and update UI
            state.selectedImages.clear();
            updateActionBar();

            showNotification(`Successfully rotated ${result.success.length} images`, "success");
        }
    } catch (error) {
        console.error("Rotate error:", error);
        showNotification(`Failed to rotate images: ${error.message}`, "error");
    } finally {
        if (notification) notification.remove();
    }
}

async function rotateImage(direction) {
    if (!state.currentImage) return;
    await rotateSingleImage(state.currentImage.path, direction);
    resetImagePosition();
}

function refreshImageInUI(imagePath) {
    // Refresh in gallery
    const card = document.querySelector(`.photo-card[data-image-path="${imagePath}"]`);
    if (card) {
        const img = card.querySelector(".photo-img");
        if (img) {
            img.src = `/images/${imagePath}?t=${Date.now()}`;
        }
    }

    // Refresh in modal if currently open
    if (state.currentImage && state.currentImage.path === imagePath) {
        elements.largeImage.src = `/bigimages/${imagePath}?t=${Date.now()}`;
    }
}

// Deletion Handling
function showSingleDeleteConfirmation(imagePath) {
    if (!elements.deleteConfirmationModalElement) return;

    // Clear any pending deletions
    state.pendingDeletion.clear();
    state.pendingDeletion.add(imagePath);

    // Update UI for single delete
    elements.deleteCount.textContent = 1;
    elements.confirmationThumbnails.innerHTML = '';

    const fileName = imagePath.split('/').pop();
    const thumbnailItem = document.createElement('div');
    thumbnailItem.className = 'thumbnail-item selected';
    thumbnailItem.dataset.imagePath = imagePath;

    const img = document.createElement('img');
    img.className = 'thumbnail-img';
    img.src = `/images/${imagePath}?t=${Date.now()}`;
    img.alt = fileName;

    thumbnailItem.appendChild(img);
    elements.confirmationThumbnails.appendChild(thumbnailItem);

    // Update modal content
    const modalBody = elements.deleteConfirmationModalElement.querySelector('.modal-body');
    if (modalBody) {
        modalBody.innerHTML = '<p>Are you sure you want to delete this image?</p>';
        modalBody.appendChild(elements.confirmationThumbnails);
    }

    // Hide select all checkbox
    const confirmCheckboxContainer = elements.deleteConfirmationModalElement.querySelector('.confirm-checkbox-container');
    if (confirmCheckboxContainer) {
        confirmCheckboxContainer.style.display = 'none';
    }

    elements.confirmDeleteBtn.disabled = false;
    elements.deleteConfirmationModal?.show();
}

function showDeleteConfirmation() {
    if (state.selectedImages.size === 0) return;

    // Clear any pending deletions and add current selection
    state.pendingDeletion.clear();
    state.selectedImages.forEach(img => state.pendingDeletion.add(img));

    elements.deleteCount.textContent = state.pendingDeletion.size;
    elements.confirmationThumbnails.innerHTML = "";

    state.pendingDeletion.forEach((imagePath) => {
        const fileName = imagePath.split("/").pop();
        const thumbnailItem = document.createElement("div");
        thumbnailItem.className = "thumbnail-item selected";
        thumbnailItem.dataset.imagePath = imagePath;

        const img = document.createElement("img");
        img.className = "thumbnail-img";
        img.src = `/images/${imagePath}?t=${Date.now()}`;
        img.alt = fileName;

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "thumbnail-checkbox";
        checkbox.checked = true;
        const checkboxId = `thumb-check-${imagePath.replace(/[^a-z0-9]/gi, "-")}`;
        checkbox.id = checkboxId;

        const label = document.createElement("label");
        label.className = "checkbox-label";
        label.htmlFor = checkboxId;

        checkbox.addEventListener("change", (e) => {
            thumbnailItem.classList.toggle("selected", e.target.checked);
            if (!e.target.checked) {
                state.pendingDeletion.delete(imagePath);
            } else {
                state.pendingDeletion.add(imagePath);
            }
            elements.deleteCount.textContent = state.pendingDeletion.size;
        });

        thumbnailItem.appendChild(img);
        thumbnailItem.appendChild(checkbox);
        thumbnailItem.appendChild(label);
        elements.confirmationThumbnails.appendChild(thumbnailItem);
    });

    // Show select all checkbox
    const confirmCheckboxContainer = elements.deleteConfirmationModalElement.querySelector('.confirm-checkbox-container');
    if (confirmCheckboxContainer) {
        confirmCheckboxContainer.style.display = 'block';
    }

    elements.confirmDeleteBtn.disabled = false;
    elements.deleteConfirmationModal?.show();
}

function handleConfirmedDelete() {
    if (state.pendingDeletion.size === 0) return;

    if (state.pendingDeletion.size === 1) {
        // Single image deletion
        const [imagePath] = state.pendingDeletion.values();
        deleteSingleImage(imagePath);
    } else {
        // Multiple images deletion
        deleteSelectedImages();
    }

    elements.deleteConfirmationModal?.hide();
}

async function deleteSingleImage(imagePath) {
    if (state.pendingDeletion.has(imagePath)) {
        const notification = showNotification(`Deleting image...`, "info", 0);
        try {
            const response = await fetch("/delete-images", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ images: [imagePath] }),
            });

            const result = await response.json();
            if (result.message) {
                removeImageFromUI(imagePath);

                if (notification) {
                    updateNotification(notification, "success", "Image deleted");
                }
            } else {
                throw new Error(result.error || "Deletion failed on server");
            }
        } catch (error) {
            console.error("Delete error:", error);
            if (notification) {
                updateNotification(notification, "error", "Deletion failed");
            }
        } finally {
            state.pendingDeletion.delete(imagePath);
            state.selectedImages.delete(imagePath);
            updateActionBar();
        }
    }
}

async function deleteSelectedImages() {
    if (state.pendingDeletion.size === 0) return;

    const imagesToDelete = Array.from(state.pendingDeletion);
    const numToDelete = imagesToDelete.length;
    const notification = showNotification(
        `Deleting ${numToDelete} images...`,
        "info",
        0
    );

    try {
        const response = await fetch("/delete-images", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ images: imagesToDelete }),
        });

        const result = await response.json();
        if (result.message) {
            imagesToDelete.forEach((imagePath) => {
                removeImageFromUI(imagePath);
            });

            if (notification) {
                updateNotification(notification, "success", result.message);
            }
        } else {
            throw new Error(result.error || "Deletion failed on server");
        }
    } catch (error) {
        console.error("Delete error:", error);
        if (notification) {
            updateNotification(notification, "error", "Deletion failed");
        }
    } finally {
        state.pendingDeletion.clear();
        state.selectedImages.clear();
        updateActionBar();
    }
}

function removeImageFromUI(imagePath) {
    const card = elements.results.querySelector(`.photo-card[data-image-path="${imagePath}"]`);
    card?.remove();

    // If the image is currently open in the modal, close the modal
    if (state.currentImage && state.currentImage.path === imagePath) {
        elements.imageModal?.hide();
        state.currentImage = null;
    }
}

// Utility Functions
function copyImagePathToClipboard(imagePath) {
    const pathToCopy = `/images/${imagePath}`;

    if (navigator.clipboard) {
        navigator.clipboard.writeText(pathToCopy)
            .then(() => {
                showNotification(`Image path copied to clipboard: ${pathToCopy}`, 'success');
            })
            .catch(err => {
                console.error('Failed to copy image path: ', err);
                showNotification('Failed to copy image path', 'error');
            });
    } else {
        // Fallback for browsers that don't support navigator.clipboard
        const tempInput = document.createElement('input');
        tempInput.value = pathToCopy;
        document.body.appendChild(tempInput);
        tempInput.select();
        document.execCommand('copy');
        document.body.removeChild(tempInput);
        showNotification(`Image path copied to clipboard: ${pathToCopy}`, 'success');
    }
}

function openImageModal(imagePath, fileName) {
    resetModalState();
    state.currentImage = { path: imagePath, name: fileName };
    elements.largeImage.src = `/bigimages/${imagePath}?t=${Date.now()}`;
    elements.imageModal.show();
}

function downloadImage() {
    if (!state.currentImage) return;
    downloadSingleImage(state.currentImage.path);
}

function downloadSingleImage(imagePath) {
    const link = document.createElement("a");
    link.href = `/bigimages/${imagePath}?t=${Date.now()}`;
    link.download = imagePath.split("/").pop() || "image";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && elements.previewImage) {
        const reader = new FileReader();
        reader.onload = (event) => {
            elements.previewImage.src = event.target.result;
            elements.previewImage.style.display = "block";
        };
        elements.previewImage.style.display = "block";
        elements.closeButton.style.display = "block";
        reader.readAsDataURL(file);
    } else if (elements.previewImage) {
        elements.previewImage.style.display = "none";
        elements.previewImage.src = "";
        elements.closeButton.style.display = "none";
    }
}

function handleClearSearchImage(e) {
    elements.previewImage.style.display = "none";
    elements.previewImage.src = "";
    elements.queryImage.files[0] = null;
    elements.closeButton.style.display = "none";
}

function handleScroll() {
    if (window.scrollY > 100 && state.selectedImages.size > 0) {
        elements.actionBar?.classList.add("visible");
    } else if (state.selectedImages.size === 0) {
        elements.actionBar?.classList.remove("visible");
    }
}

function toggleLoading(show) {
    state.isLoading = show;
    if (elements.loading) {
        elements.loading.style.display = show ? "flex" : "none";
    }
}

function showNotification(message, type = "info", duration = 3000) {
    if (!elements.notificationContainer) return null;

    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas ${getNotificationIcon(type)} me-2"></i>
        ${message}
    `;

    elements.notificationContainer.appendChild(notification);

    if (duration > 0) {
        setTimeout(() => {
            notification.style.animation = "slideIn 0.3s ease-out reverse";
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }

    return notification;
}

function updateNotification(notification, type, message) {
    if (!notification) return;

    notification.classList.remove("info", "error", "success", "warning");
    notification.classList.add(type);
    notification.innerHTML = `
        <i class="fas ${getNotificationIcon(type)} me-2"></i>
        ${message}
    `;

    setTimeout(() => notification.remove(), 3000);
}

function getNotificationIcon(type) {
    const icons = {
        success: "fa-check-circle",
        error: "fa-exclamation-circle",
        warning: "fa-exclamation-triangle",
        info: "fa-info-circle",
    };
    return icons[type] || "fa-info-circle";
}