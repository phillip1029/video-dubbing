// Video Dubbing Application JavaScript

class VideoDubbingApp {
    constructor() {
        this.socket = null;
        this.currentSessionId = null;
        this.supportedLanguages = {};
        
        this.init();
    }
    
    init() {
        this.setupSocketIO();
        this.setupEventListeners();
        this.loadConfiguration();
    }
    
    setupSocketIO() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.showNotification('Connected to server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.showNotification('Disconnected from server', 'warning');
        });
        
        this.socket.on('status_update', (data) => {
            this.updateStatus(data);
        });
        
        this.socket.on('processing_complete', (data) => {
            console.log('Processing complete:', data);
            this.handleProcessingComplete(data);
        });
        
        this.socket.on('processing_error', (data) => {
            console.log('Processing error:', data);
            this.handleProcessingError(data);
        });
    }
    
    setupEventListeners() {
        // Upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleUpload();
        });
        
        // Reference audio upload
        document.getElementById('referenceAudio').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.uploadReferenceAudio(e.target.files[0]);
            }
        });
        
        // Download button
        document.getElementById('downloadButton').addEventListener('click', () => {
            this.downloadResult();
        });
        
        // Review approval
        document.getElementById('approveAllButton').addEventListener('click', () => {
            this.approveAllTranslations();
        });
        
        // File drag and drop
        this.setupDragAndDrop();
    }
    
    setupDragAndDrop() {
        const videoInput = document.getElementById('videoFile');
        const uploadForm = document.getElementById('uploadForm');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadForm.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadForm.addEventListener(eventName, () => {
                uploadForm.classList.add('dragover');
            });
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadForm.addEventListener(eventName, () => {
                uploadForm.classList.remove('dragover');
            });
        });
        
        uploadForm.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                videoInput.files = files;
                this.showNotification('Video file added', 'success');
            }
        });
    }
    
    async loadConfiguration() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            
            this.supportedLanguages = config.supported_languages;
            this.populateLanguageSelect();
            
        } catch (error) {
            console.error('Failed to load configuration:', error);
            this.showNotification('Failed to load configuration', 'error');
        }
    }
    
    populateLanguageSelect() {
        const select = document.getElementById('targetLanguage');
        
        for (const [code, name] of Object.entries(this.supportedLanguages)) {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = name;
            select.appendChild(option);
        }
    }
    
    async handleUpload() {
        const form = document.getElementById('uploadForm');
        const videoFile = document.getElementById('videoFile').files[0];
        const targetLanguage = document.getElementById('targetLanguage').value;
        const autoApprove = document.getElementById('autoApprove').checked;
        
        if (!videoFile || !targetLanguage) {
            this.showNotification('Please select a video file and target language', 'error');
            return;
        }
        
        this.setLoading(true);
        
        try {
            // Upload video file
            const videoFormData = new FormData();
            videoFormData.append('video', videoFile);
            
            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                body: videoFormData
            });
            
            const uploadResult = await uploadResponse.json();
            
            if (!uploadResult.success) {
                throw new Error(uploadResult.error);
            }
            
            // Start processing
            const processData = {
                video_path: uploadResult.file_path,
                target_language: targetLanguage,
                auto_approve: autoApprove
            };
            
            // Add reference audio if uploaded
            if (this.referenceAudioPath) {
                processData.speaker_reference = this.referenceAudioPath;
            }
            
            const processResponse = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(processData)
            });
            
            const processResult = await processResponse.json();
            
            if (!processResult.success) {
                throw new Error(processResult.error);
            }
            
            this.currentSessionId = processResult.session_id;
            this.showProcessingInterface();
            this.subscribeToStatusUpdates();
            
            this.showNotification('Processing started successfully', 'success');
            
        } catch (error) {
            console.error('Upload failed:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    async uploadReferenceAudio(file) {
        try {
            const formData = new FormData();
            formData.append('audio', file);
            
            const response = await fetch('/api/upload/reference', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.referenceAudioPath = result.file_path;
                this.showNotification('Reference audio uploaded', 'success');
            } else {
                throw new Error(result.error);
            }
            
        } catch (error) {
            console.error('Reference audio upload failed:', error);
            this.showNotification(`Reference audio upload failed: ${error.message}`, 'error');
        }
    }
    
    showProcessingInterface() {
        document.getElementById('initialState').classList.add('d-none');
        document.getElementById('statusContainer').classList.remove('d-none');
        document.getElementById('sessionId').textContent = this.currentSessionId;
        
        this.initializeStageProgress();
    }
    
    initializeStageProgress() {
        const stages = [
            { key: 'asr', name: 'Speech Recognition', icon: 'microphone' },
            { key: 'translation', name: 'Translation', icon: 'language' },
            { key: 'review', name: 'Review', icon: 'edit' },
            { key: 'tts', name: 'Speech Synthesis', icon: 'volume-up' },
            { key: 'lipsync', name: 'Lip Synchronization', icon: 'video' },
            { key: 'finalize', name: 'Finalization', icon: 'check' }
        ];
        
        const container = document.getElementById('stageProgress');
        container.innerHTML = '';
        
        stages.forEach(stage => {
            const stageElement = document.createElement('div');
            stageElement.className = 'stage-item';
            stageElement.innerHTML = `
                <div class="stage-icon pending" id="icon-${stage.key}">
                    <i class="fas fa-${stage.icon}"></i>
                </div>
                <div class="stage-content">
                    <div class="stage-name">${stage.name}</div>
                    <div class="stage-progress" id="progress-${stage.key}">Pending</div>
                </div>
            `;
            container.appendChild(stageElement);
        });
    }
    
    subscribeToStatusUpdates() {
        this.socket.emit('subscribe_status', {
            session_id: this.currentSessionId
        });
    }
    
    updateStatus(statusData) {
        // Update overall progress
        const progress = Math.round(statusData.progress * 100);
        document.getElementById('overallProgress').textContent = `${progress}%`;
        document.getElementById('overallProgressBar').style.width = `${progress}%`;
        
        // Update current stage
        const currentStageElement = document.getElementById('currentStage');
        const stageName = this.getStageDisplayName(statusData.current_stage);
        const stageStatus = statusData.stages[statusData.current_stage]?.status || 'pending';
        
        currentStageElement.innerHTML = `
            <i class="fas fa-${this.getStageIcon(stageStatus)} me-2"></i>
            ${stageName}: ${this.capitalizeFirst(stageStatus)}
        `;
        
        // Update individual stage progress
        Object.entries(statusData.stages).forEach(([stageKey, stageData]) => {
            this.updateStageProgress(stageKey, stageData);
        });
        
        // Handle completed status
        if (statusData.status === 'completed') {
            this.handleProcessingComplete({ session_id: this.currentSessionId });
        }
    }
    
    updateStageProgress(stageKey, stageData) {
        const iconElement = document.getElementById(`icon-${stageKey}`);
        const progressElement = document.getElementById(`progress-${stageKey}`);
        
        if (!iconElement || !progressElement) return;
        
        // Update icon
        iconElement.className = `stage-icon ${stageData.status}`;
        
        // Update progress text
        const progressPercent = Math.round(stageData.progress * 100);
        progressElement.textContent = `${this.capitalizeFirst(stageData.status)} (${progressPercent}%)`;
    }
    
    getStageDisplayName(stageKey) {
        const stageNames = {
            'asr': 'Speech Recognition',
            'translation': 'Translation',
            'review': 'Review',
            'tts': 'Speech Synthesis',
            'lipsync': 'Lip Synchronization',
            'finalize': 'Finalization'
        };
        return stageNames[stageKey] || stageKey;
    }
    
    getStageIcon(status) {
        const icons = {
            'pending': 'clock',
            'running': 'spinner fa-spin',
            'completed': 'check',
            'failed': 'times'
        };
        return icons[status] || 'clock';
    }
    
    handleProcessingComplete(data) {
        document.getElementById('currentStage').innerHTML = `
            <i class="fas fa-check-circle me-2"></i>
            Processing completed successfully!
        `;
        
        document.getElementById('downloadSection').classList.remove('d-none');
        
        this.showNotification('Video dubbing completed!', 'success');
    }
    
    handleProcessingError(data) {
        document.getElementById('errorSection').classList.remove('d-none');
        document.getElementById('errorMessage').textContent = data.error;
        
        this.showNotification(`Processing failed: ${data.error}`, 'error');
    }
    
    async downloadResult() {
        if (!this.currentSessionId) return;
        
        try {
            const response = await fetch(`/api/download/${this.currentSessionId}`);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error);
            }
            
            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dubbed_video_${this.currentSessionId}.mp4`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            this.showNotification('Download started', 'success');
            
        } catch (error) {
            console.error('Download failed:', error);
            this.showNotification(`Download failed: ${error.message}`, 'error');
        }
    }
    
    async loadReviewData() {
        if (!this.currentSessionId) return;
        
        try {
            const response = await fetch(`/api/review/${this.currentSessionId}`);
            const reviewData = await response.json();
            
            this.displayReviewInterface(reviewData);
            
        } catch (error) {
            console.error('Failed to load review data:', error);
            this.showNotification('Failed to load review data', 'error');
        }
    }
    
    displayReviewInterface(reviewData) {
        const container = document.getElementById('reviewContent');
        container.innerHTML = '';
        
        reviewData.segments.forEach((segment, index) => {
            const segmentElement = document.createElement('div');
            segmentElement.className = `review-segment ${segment.approved ? 'approved' : ''} ${segment.edited ? 'edited' : ''}`;
            segmentElement.innerHTML = `
                <div class="segment-timing">
                    ${this.formatTime(segment.start)} - ${this.formatTime(segment.end)}
                </div>
                <div class="original-text">
                    <strong>Original:</strong> ${segment.original_text}
                </div>
                <div class="mb-2">
                    <label class="form-label"><strong>Translation:</strong></label>
                    <textarea class="form-control translation-input" 
                              data-segment-id="${segment.id}"
                              rows="3">${segment.translated_text}</textarea>
                </div>
                <div class="segment-controls">
                    <button class="btn btn-sm btn-success approve-btn" 
                            data-segment-id="${segment.id}"
                            ${segment.approved ? 'disabled' : ''}>
                        <i class="fas fa-check me-1"></i>
                        ${segment.approved ? 'Approved' : 'Approve'}
                    </button>
                    <button class="btn btn-sm btn-primary save-btn" 
                            data-segment-id="${segment.id}">
                        <i class="fas fa-save me-1"></i>
                        Save Changes
                    </button>
                </div>
            `;
            container.appendChild(segmentElement);
        });
        
        // Add event listeners for review actions
        this.setupReviewEventListeners();
        
        // Show review modal
        const reviewModal = new bootstrap.Modal(document.getElementById('reviewModal'));
        reviewModal.show();
    }
    
    setupReviewEventListeners() {
        // Approve buttons
        document.querySelectorAll('.approve-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const segmentId = parseInt(e.target.dataset.segmentId);
                this.approveSegment(segmentId);
            });
        });
        
        // Save buttons
        document.querySelectorAll('.save-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const segmentId = parseInt(e.target.dataset.segmentId);
                this.saveSegmentTranslation(segmentId);
            });
        });
    }
    
    async approveSegment(segmentId) {
        try {
            const textarea = document.querySelector(`textarea[data-segment-id="${segmentId}"]`);
            const translation = textarea.value;
            
            const response = await fetch(`/api/review/${this.currentSessionId}/segment/${segmentId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    translation: translation,
                    approved: true
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                const approveBtn = document.querySelector(`.approve-btn[data-segment-id="${segmentId}"]`);
                approveBtn.disabled = true;
                approveBtn.innerHTML = '<i class="fas fa-check me-1"></i>Approved';
                
                const segmentElement = textarea.closest('.review-segment');
                segmentElement.classList.add('approved');
                
                this.showNotification('Segment approved', 'success');
            }
            
        } catch (error) {
            console.error('Failed to approve segment:', error);
            this.showNotification('Failed to approve segment', 'error');
        }
    }
    
    async saveSegmentTranslation(segmentId) {
        try {
            const textarea = document.querySelector(`textarea[data-segment-id="${segmentId}"]`);
            const translation = textarea.value;
            
            const response = await fetch(`/api/review/${this.currentSessionId}/segment/${segmentId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    translation: translation,
                    approved: false
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                const segmentElement = textarea.closest('.review-segment');
                segmentElement.classList.add('edited');
                
                this.showNotification('Translation saved', 'success');
            }
            
        } catch (error) {
            console.error('Failed to save translation:', error);
            this.showNotification('Failed to save translation', 'error');
        }
    }
    
    async approveAllTranslations() {
        try {
            const response = await fetch(`/api/review/${this.currentSessionId}/approve`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('All translations approved', 'success');
                
                // Close modal
                const reviewModal = bootstrap.Modal.getInstance(document.getElementById('reviewModal'));
                reviewModal.hide();
                
                // Continue processing
                this.subscribeToStatusUpdates();
            }
            
        } catch (error) {
            console.error('Failed to approve all translations:', error);
            this.showNotification('Failed to approve translations', 'error');
        }
    }
    
    setLoading(loading) {
        const startButton = document.getElementById('startButton');
        const uploadForm = document.getElementById('uploadForm');
        
        if (loading) {
            startButton.disabled = true;
            startButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            uploadForm.classList.add('loading');
        } else {
            startButton.disabled = false;
            startButton.innerHTML = '<i class="fas fa-play me-2"></i>Start Dubbing';
            uploadForm.classList.remove('loading');
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VideoDubbingApp();
});
