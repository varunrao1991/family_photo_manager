const config = {
    perPage: 30,
    infiniteScrollThreshold: 20,
    zoomStep: 0.2,
    rotateStep: 90,
    panBorderSize: 50, // Size of edge area that triggers panning
    panSensitivity: 10, // How fast the image moves
};

let state = {
    currentPage: 1,
    isLoading: false,
    hasMore: true,
    selectedImages: new Set(),
    lastSearch: null,
    zoomLevel: 1,
    currentRotation: 0,
    currentImage: null,
    isDragging: false,
    startX: 0,
    startY: 0,
    translateX: 0,
    translateY: 0,
    savedTransform: "",
    panDirection: { x: 0, y: 0 }, // Track pan direction
    panInterval: null, // For continuous panning
};

const elements = {
    searchForm: document.getElementById("searchForm"),
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
    deleteConfirmationModalElement: document.getElementById(
        "deleteConfirmationModal"
    ),
    confirmationThumbnails: document.getElementById("confirmationThumbnails"),
    deleteCount: document.getElementById("delete-count"),
    confirmDeleteCheckbox: document.getElementById("confirmDeleteCheckbox"),
    confirmDeleteBtn: document.getElementById("confirmDeleteBtn"),
    notificationContainer: document.getElementById("notificationContainer"),
    imageModal: null,
    helpModal: null,
    deleteConfirmationModal: null,
};

document.addEventListener("DOMContentLoaded", () => {
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
    setupEventListeners();
    setupIntersectionObserver();
});

function resetImagePosition() {
    if (!elements.largeImage) return;
    state.translateX = 0;
    state.translateY = 0;
    applyImageTransform();
}

function setupEventListeners() {
    document.getElementById('resetPosition')?.addEventListener('click', resetImagePosition);
    elements.searchForm?.addEventListener('submit', handleSearchSubmit);
    elements.queryImage?.addEventListener('change', handleFileSelect);
    elements.selectAll?.addEventListener('click', selectAllImages);
    elements.rotateLeftSelected?.addEventListener('click', () => rotateSelected('left'));
    elements.rotateRightSelected?.addEventListener('click', () => rotateSelected('right'));
    elements.deleteSelected?.addEventListener('click', showDeleteConfirmation);
    elements.rotateLeftModal?.addEventListener('click', () => rotateImage(-config.rotateStep));
    elements.rotateRightModal?.addEventListener('click', () => rotateImage(config.rotateStep));
    elements.downloadImage?.addEventListener('click', downloadImage);
    elements.confirmDeleteCheckbox?.addEventListener('change', (e) => {
        if (elements.confirmDeleteBtn) {
            elements.confirmDeleteBtn.disabled = !e.target.checked;
        }
    });
    elements.confirmDeleteBtn?.addEventListener('click', deleteSelected);
    elements.toggleHelp?.addEventListener('click', () => elements.helpModal?.show());
    window.addEventListener('scroll', handleScroll);
    elements.zoomInModal?.addEventListener('click', zoomInImage);
    elements.zoomOutModal?.addEventListener('click', zoomOutImage);

    // Add mouse move listener for edge panning
    if (elements.imageModalElement) {
        elements.imageModalElement.addEventListener('mousemove', handleModalMouseMove);
        elements.imageModalElement.addEventListener('mouseleave', stopEdgePan);
        elements.imageModalElement.addEventListener('hidden.bs.modal', () => {
            stopEdgePan();
            // Reset other state
            state.zoomLevel = 1;
            state.currentRotation = 0;
            state.translateX = 0;
            state.translateY = 0;
            if (elements.largeImage) {
                elements.largeImage.style.transform = '';
            }
        });
    }
}

function zoomInImage() {
    state.zoomLevel += config.zoomStep;
    applyImageTransform();
}

function zoomOutImage() {
    state.zoomLevel = Math.max(0.1, state.zoomLevel - config.zoomStep);
    applyImageTransform();
}

function handleModalMouseMove(e) {
    if (!elements.largeImage || state.isDragging) return;

    const modal = elements.imageModalElement.querySelector('.modal-dialog');
    const modalRect = modal.getBoundingClientRect();
    const imgRect = elements.largeImage.getBoundingClientRect();

    // Calculate available space around the image
    const leftSpace = imgRect.left - modalRect.left;
    const rightSpace = modalRect.right - imgRect.right;
    const topSpace = imgRect.top - modalRect.top;
    const bottomSpace = modalRect.bottom - imgRect.bottom;

    // Determine if we need to pan in each direction
    const shouldPanLeft = leftSpace < 0 && e.clientX < modalRect.left + config.panBorderSize;
    const shouldPanRight = rightSpace < 0 && e.clientX > modalRect.right - config.panBorderSize;
    const shouldPanUp = topSpace < 0 && e.clientY < modalRect.top + config.panBorderSize;
    const shouldPanDown = bottomSpace < 0 && e.clientY > modalRect.bottom - config.panBorderSize;

    // Calculate pan direction
    const panX = shouldPanLeft ? 1 : shouldPanRight ? -1 : 0;
    const panY = shouldPanUp ? 1 : shouldPanDown ? -1 : 0;

    // Only update if direction changed
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
        // Calculate movement based on direction and sensitivity
        const moveX = state.panDirection.x * config.panSensitivity;
        const moveY = state.panDirection.y * config.panSensitivity;

        // Update translation values
        state.translateX += moveX;
        state.translateY += moveY;

        // Apply the transform
        applyImageTransform();
    }, 16); // ~60fps
}

function applyImageTransform() {
    if (!elements.largeImage) return;

    elements.largeImage.style.transform = `
        translate(${state.translateX}px, ${state.translateY}px)
        rotate(${state.currentRotation}deg)
        scale(${state.zoomLevel})
    `;
}

function stopEdgePan() {
    if (state.panInterval) {
        clearInterval(state.panInterval);
        state.panInterval = null;
    }
    state.panDirection = { x: 0, y: 0 };
}

function setupIntersectionObserver() {
    if (!elements.infiniteScrollTrigger) return;
    const observer = new IntersectionObserver(
        (entries) => {
            if (
                entries[0].isIntersecting &&
                !state.isLoading &&
                state.hasMore &&
                state.lastSearch
            ) {
                loadMoreResults();
            }
        },
        { threshold: 0.1 }
    );
    observer.observe(elements.infiniteScrollTrigger);
}

async function handleSearchSubmit(e) {
    e.preventDefault();
    state.currentPage = 1;
    state.hasMore = true;
    state.selectedImages.clear();
    updateActionBar();
    elements.results.innerHTML = "";
    toggleLoading(true);
    const formData = new FormData();
    if (elements.queryText.value) {
        formData.append("query_text", elements.queryText.value);
    }
    if (elements.queryImage.files[0]) {
        formData.append("query_image", elements.queryImage.files[0]);
    }
    formData.append("page", state.currentPage);
    formData.append("per_page", config.perPage);
    state.lastSearch = formData;
    try {
        const response = await fetch("/similar-images", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        displayResults(data.similar_images || []);
        state.hasMore = data.similar_images
            ? data.similar_images.length === config.perPage
            : false;
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
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
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
    if (images.length === 0 && !append && state.lastSearch) {
        elements.results.innerHTML = `<div class="col-12 text-center py-5"><i class="fas fa-image fa-3x mb-3 text-muted"></i><h5 class="text-muted">No more images found for this search</h5></div>`;
        state.hasMore = false;
        return;
    } else if (images.length === 0 && !append && !state.lastSearch) {
        elements.results.innerHTML = `<div class="col-12 text-center py-5"><i class="fas fa-image fa-3x mb-3 text-muted"></i><h5 class="text-muted">No images found</h5><p>Try a different search term or upload another image</p></div>`;
        return;
    }
    images.forEach((imagePath) => {
        const fileName = imagePath.split("/").pop();
        const card = document.createElement("div");
        card.className = "photo-card";
        card.dataset.imagePath = imagePath;
        if (state.selectedImages.has(imagePath)) {
            card.classList.add("selected");
        }
        const img = document.createElement("img");
        img.className = "photo-img";
        img.src = `/images/${imagePath}`;
        img.loading = "lazy";
        img.alt = fileName;
        const overlay = document.createElement("div");
        overlay.className = "photo-overlay";
        overlay.textContent = fileName;
        const actions = document.createElement("div");
        actions.className = "photo-actions";
        const rotateLeftBtn = document.createElement("button");
        rotateLeftBtn.className = "photo-action-btn";
        rotateLeftBtn.innerHTML = '<i class="fas fa-undo"></i>';
        rotateLeftBtn.title = "Rotate Left";
        rotateLeftBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            rotateSingleImage(imagePath, "left");
        });
        const rotateRightBtn = document.createElement("button");
        rotateRightBtn.className = "photo-action-btn";
        rotateRightBtn.innerHTML = '<i class="fas fa-redo"></i>';
        rotateRightBtn.title = "Rotate Right";
        rotateRightBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            rotateSingleImage(imagePath, "right");
        });
        const copyPathBtn = document.createElement("button");
        copyPathBtn.className = "photo-action-btn";
        copyPathBtn.innerHTML = '<i class="fas fa-link"></i>';
        copyPathBtn.title = "Copy Path";
        copyPathBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            copyImagePathToClipboard(imagePath);
        });
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "photo-action-btn";
        deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
        deleteBtn.title = "Delete";
        deleteBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            showSingleDeleteConfirmation(imagePath);
        });
        actions.appendChild(rotateLeftBtn);
        actions.appendChild(rotateRightBtn);
        actions.appendChild(copyPathBtn); // Add the copy path button
        actions.appendChild(deleteBtn);
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
    });
}

async function deleteSingleImage(imagePath) {
    const notification = showNotification(`Deleting image...`, "info", 0);
    try {
        const response = await fetch("/delete-images", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ images: [imagePath] }),
        });
        const result = await response.json();
        if (result.message) {
            const cardToRemove = elements.results.querySelector(
                `.photo-card[data-image-path="${imagePath}"]`
            );
            cardToRemove?.remove();
            if (notification) {
                notification.classList.remove("info");
                notification.classList.add("success");
                notification.innerHTML = `<i class="fas fa-check-circle"></i> Image deleted`;
                setTimeout(() => notification.remove(), 3000);
            }
            state.selectedImages.delete(imagePath); // Remove from selected if it was
            updateActionBar();
        } else {
            throw new Error(result.error || "Deletion failed on server");
        }
    } catch (error) {
        console.error("Delete error:", error);
        if (notification) {
            notification.classList.remove("info");
            notification.classList.add("error");
            notification.innerHTML = `<i class="fas fa-exclamation-circle"></i> Deletion failed`;
            setTimeout(() => notification.remove(), 3000);
        }
    }
}

function showSingleDeleteConfirmation(imagePath) {
    if (!elements.deleteConfirmationModalElement) return; // Safety check

    elements.deleteCount.textContent = 1; // Always 1 for single delete
    elements.confirmationThumbnails.innerHTML = ''; // Clear previous thumbnails

    // Create thumbnail for single image
    const fileName = imagePath.split('/').pop();
    const thumbnailItem = document.createElement('div');
    thumbnailItem.className = 'thumbnail-item selected';
    thumbnailItem.dataset.imagePath = imagePath;

    const img = document.createElement('img');
    img.className = 'thumbnail-img';
    img.src = `/images/${imagePath}`;
    img.alt = fileName;

    thumbnailItem.appendChild(img);
    elements.confirmationThumbnails.appendChild(thumbnailItem);

    // Update modal content for single delete message
    const modalBody = elements.deleteConfirmationModalElement.querySelector('.modal-body');
    if (modalBody) {
        modalBody.innerHTML = '<p>Are you sure you want to delete this image?</p>';
        modalBody.appendChild(elements.confirmationThumbnails); // Add thumbnails
    }

    // Hide the "select all" checkbox and associated UI
    const confirmCheckboxContainer = elements.deleteConfirmationModalElement.querySelector('.confirm-checkbox-container');
    if (confirmCheckboxContainer) {
        confirmCheckboxContainer.style.display = 'none';
    }

    // Update confirm button action
    elements.confirmDeleteBtn.onclick = () => {
        deleteSingleImage(imagePath);
        elements.deleteConfirmationModal?.hide();
    };
    elements.confirmDeleteBtn.disabled = false; // Enable the button

    elements.deleteConfirmationModal?.show();
}

function copyImagePathToClipboard(imagePath) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(`/images/${imagePath}`)
            .then(() => {
                showNotification(`Image path copied to clipboard: /images/${imagePath}`, 'success');
            })
            .catch(err => {
                console.error('Failed to copy image path: ', err);
                showNotification('Failed to copy image path', 'error');
            });
    } else {
        // Fallback for browsers that don't support navigator.clipboard
        const tempInput = document.createElement('input');
        tempInput.value = `/images/${imagePath}`;
        document.body.appendChild(tempInput);
        tempInput.select();
        document.execCommand('copy');
        document.body.removeChild(tempInput);
        showNotification(`Image path copied to clipboard: /images/${imagePath}`, 'success');
    }
}

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
        allCards.forEach((card) => card.classList.remove("selected"));
        state.selectedImages.clear();
    } else {
        allCards.forEach((card) => {
            if (
                card.dataset.imagePath &&
                !state.selectedImages.has(card.dataset.imagePath)
            ) {
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
        if (elements.rotateLeftSelected) elements.rotateLeftSelected.disabled = false;
        if (elements.rotateRightSelected)
            elements.rotateRightSelected.disabled = false;
        if (elements.deleteSelected) elements.deleteSelected.disabled = false;
    } else {
        elements.actionBar?.classList.remove("visible");
        if (elements.selectedCount) {
            elements.selectedCount.classList.add("d-none");
        }
        if (elements.rotateLeftSelected) elements.rotateLeftSelected.disabled = true;
        if (elements.rotateRightSelected)
            elements.rotateRightSelected.disabled = true;
        if (elements.deleteSelected) elements.deleteSelected.disabled = true;
    }
    if (elements.selectAll) {
        elements.selectAll.textContent =
            state.selectedImages.size === document.querySelectorAll(".photo-card").length &&
                document.querySelectorAll(".photo-card").length > 0
                ? "Deselect All"
                : "Select All";
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
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        if (result.success) {
            state.selectedImages.clear();
            updateActionBar();
            if (state.lastSearch) {
                const refreshFormData = state.lastSearch;
                refreshFormData.set("page", 1);
                state.lastSearch = refreshFormData;
                state.currentPage = 1;
                state.hasMore = true

                elements.results.innerHTML = "";
                toggleLoading(true);
                try {
                    const refreshResponse = await fetch("/similar-images", {
                        method: "POST",
                        body: refreshFormData,
                    });
                    const refreshData = await refreshResponse.json();
                    displayResults(refreshData.similar_images || []);
                } catch (refreshError) {
                    console.error("Refresh after rotate error:", refreshError);
                } finally {
                    toggleLoading(false);
                }
            } else {
                elements.results.innerHTML = "";
            }
        }
    } catch (error) {
        console.error("Rotate error:", error);
    }
}

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
            const card = document.querySelector(
                `.photo-card[data-image-path="${imagePath}"]`
            );
            if (card) {
                const img = card.querySelector(".photo-img");
                if (img) {
                    img.src = `/images/${imagePath}?t=${Date.now()}`;
                }
            }
            if (notification) {
                notification.classList.remove("info");
                notification.classList.add("success");
                notification.innerHTML = `<i class="fas fa-check-circle"></i> Image rotated`;
                setTimeout(() => notification.remove(), 3000);
            }
        } else {
            throw new Error(result.error || "Rotation failed on server");
        }
    } catch (error) {
        console.error("Rotate error:", error);
        if (notification) {
            notification.classList.remove("info");
            notification.classList.add("error");
            notification.innerHTML = `<i class="fas fa-exclamation-circle"></i> Rotation failed`;
            setTimeout(() => notification.remove(), 3000);
        }
    }
}

function showDeleteConfirmation() {
    if (state.selectedImages.size === 0) return;
    elements.deleteCount.textContent = state.selectedImages.size;
    elements.confirmationThumbnails.innerHTML = "";
    state.selectedImages.forEach((imagePath) => {
        const fileName = imagePath.split("/").pop();
        const thumbnailItem = document.createElement("div");
        thumbnailItem.className = "thumbnail-item selected";
        thumbnailItem.dataset.imagePath = imagePath;
        const img = document.createElement("img");
        img.className = "thumbnail-img";
        img.src = `/images/${imagePath}`;
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
                state.selectedImages.delete(imagePath);
            } else {
                state.selectedImages.add(imagePath);
            }
            elements.deleteCount.textContent = state.selectedImages.size;
        });
        thumbnailItem.appendChild(img);
        thumbnailItem.appendChild(checkbox);
        thumbnailItem.appendChild(label);
        elements.confirmationThumbnails.appendChild(thumbnailItem);
    });
    elements.confirmDeleteCheckbox.checked = false;
    elements.confirmDeleteBtn.disabled = true;
    elements.deleteConfirmationModal?.show();
}

async function deleteSelected() {
    if (state.selectedImages.size === 0) return;
    const imagesToDelete = Array.from(state.selectedImages);
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
                const card = elements.results.querySelector(
                    `.photo-card[data-image-path="${imagePath}"]`
                );
                card?.remove();
            });
            if (notification) {
                notification.classList.remove("info");
                notification.classList.add("success");
                notification.innerHTML = `<i class="fas fa-check-circle"></i> Deleted ${numToDelete} images`;
                setTimeout(() => notification.remove(), 3000);
            }
            state.selectedImages.clear();
            updateActionBar();
            elements.deleteConfirmationModal?.hide();
        } else {
            throw new Error(result.error || "Deletion failed on server");
        }
    } catch (error) {
        console.error("Delete error:", error);
        if (notification) {
            notification.classList.remove("info");
            notification.classList.add("error");
            notification.innerHTML = `<i class="fas fa-exclamation-circle"></i> Deletion failed`;
            setTimeout(() => notification.remove(), 3000);
        }
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

function getNotificationIcon(type) {
    const icons = {
        success: "fa-check-circle",
        error: "fa-exclamation-circle",
        warning: "fa-exclamation-triangle",
        info: "fa-info-circle",
    };
    return icons[type] || "fa-info-circle";
}

function openImageModal(imagePath, fileName) {
    // Reset all transform state
    state.translateX = 0;
    state.translateY = 0;
    state.zoomLevel = 1;
    state.currentRotation = 0;
    state.isDragging = false;

    // Set the new image
    state.currentImage = { path: imagePath, name: fileName };
    elements.largeImage.src = `/bigimages/${imagePath}`;
    elements.largeImage.style.transform = '';

    elements.imageModal.show();
}


function rotateImage(degrees) {
    state.currentRotation = (state.currentRotation + degrees) % 360;
    applyImageTransform();
}

function downloadImage() {
    if (!state.currentImage) return;
    const link = document.createElement("a");
    link.href = `/bigimages/${state.currentImage.path}`;
    link.download = state.currentImage.name || "image";
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
        reader.readAsDataURL(file);
    } else if (elements.previewImage) {
        elements.previewImage.style.display = "none";
        elements.previewImage.src = "";
    }
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